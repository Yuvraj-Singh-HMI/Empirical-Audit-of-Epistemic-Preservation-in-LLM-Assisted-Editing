import pandas as pd
import torch
import torch.nn.functional as F
import gc
import re
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

def compute_lexical_baselines_native(df):
    print("Initializing native BLEU and ROUGE scorers...")
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    chencherry = SmoothingFunction().method1

    preds = df["hypothesis_text"].tolist()
    refs = df["reference_text"].tolist()

    rouge_scores = []
    bleu_scores = []

    print("Computing sentence-level ROUGE-L and BLEU...")
    for hyp, ref in zip(preds, refs):
        # 1. ROUGE-L (F-measure)
        r_score = rouge.score(ref, hyp)['rougeL'].fmeasure
        rouge_scores.append(r_score)

        # 2. BLEU
        ref_tokens = nltk.word_tokenize(ref)
        hyp_tokens = nltk.word_tokenize(hyp)
        b_score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=chencherry)
        bleu_scores.append(b_score)

    df["score_rougeL"] = rouge_scores
    df["score_bleu"] = bleu_scores
    return df


def compute_semantic_baseline(df):
    print("Initializing BERTScore (microsoft/deberta-xlarge-mnli)...")
    preds = df["hypothesis_text"].tolist()
    refs = df["reference_text"].tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    P, R, F1 = score(preds, refs, model_type="microsoft/deberta-xlarge-mnli", lang="en", verbose=True, device=device)

    df["score_bertscore"] = F1.cpu().numpy()

    torch.cuda.empty_cache()
    gc.collect()

    return df


def compute_epistemic_shift(df):
    print("Loading Asymmetrical NLI Model (cross-encoder/nli-deberta-v3-large)...")
    model_name = "cross-encoder/nli-deberta-v3-large"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    def get_entailment_prob(premise, hypothesis):
        inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        return probs[0][1].item()

    esm_scores = []

    print("Computing Forward and Backward Entailments...")
    for idx, row in df.iterrows():
        s_orig = row["reference_text"]
        s_edit = row["hypothesis_text"]

        p_orig_given_edit = get_entailment_prob(premise=s_edit, hypothesis=s_orig)
        p_edit_given_orig = get_entailment_prob(premise=s_orig, hypothesis=s_edit)

        delta_epistemic = p_orig_given_edit - p_edit_given_orig
        esm_scores.append(delta_epistemic)

    df["score_delta_epistemic"] = esm_scores

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return df


def compute_epistemic_shift_llm(df):
    print("Loading Generative NLI Model (Qwen/Qwen2.5-1.5B-Instruct in fp16)...")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()

    def get_entailment_prob(premise, hypothesis):
        prompt = (
            f"Premise: '{premise}'\n"
            f"Hypothesis: '{hypothesis}'\n\n"
            "Rule 1: A correlational finding (e.g., 'associated with') does NOT entail a causal finding.\n"
            "Rule 2: A causal finding (e.g., 'causes', 'contributes to') logically ENTAILS a correlational finding.\n\n"
            "Task: Based strictly on these rules, what is the probability (from 0.0 to 1.0) that the Premise entails the Hypothesis?\n"
            "Output only the numeric float."
        )

        messages = [
            {"role": "system", "content": "You are a strict, logical biomedical text analyzer."},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        match = re.search(r"0\.\d+|1\.0|0", response)
        if match:
            return float(match.group())
        return 0.5 

    esm_scores = []

    print("Computing Forward and Backward Entailments via LLM Generation...")
    for idx, row in df.iterrows():
        s_orig = row["reference_text"]
        s_edit = row["hypothesis_text"]

        p_orig_given_edit = get_entailment_prob(premise=s_edit, hypothesis=s_orig)
        p_edit_given_orig = get_entailment_prob(premise=s_orig, hypothesis=s_edit)

        delta_epistemic = p_orig_given_edit - p_edit_given_orig
        esm_scores.append(delta_epistemic)

    df["score_delta_epistemic_llm"] = esm_scores

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return df
