# AI-Use and Research-Integrity Report
**Project:** Empirical Audit of Automated Metrics for Epistemic Preservation in LLM-Assisted Editing
**Author:** Yuvraj Singh Rajawat

## 1. Overview of AI Assistance
This project utilized Large Language Models (LLMs) in two distinct capacities:
1. **As an Experimental Subject/Tool:** ChatGPT was used to programmatically generate the synthetic evaluation dataset (paraphrases and escalations). A local generative LLM (`Qwen2.5-1.5B-Instruct`) was used to compute the proposed $\Delta_{\text{epistemic}}$ metric (Pathway C).
2. **As a Research Assistant (Gemini):** Used for code generation, pipeline orchestration, qualitative error analysis, statistical bootstrapping, and manuscript drafting.

---

## 2. Significant AI-Assisted Activities Log

### Activity 1: Dataset Generation (Triplets)
* **Task:** Create a 200-pair diagnostic dataset of epistemic shifts based on real PubMed claims.
* **Tool:** ChatGPT (via user prompt).
* **Prompt/Version:** "You are an expert biomedical linguistic annotator... strictly preserve correlational... upgrade correlational verb into a DIRECT CAUSAL verb..."
* **AI Proposal:** Generated 100 triplets (Original, Paraphrase, Escalated).
* **Human Decision & Verification:** The user accepted the dataset and fed it into the evaluation pipeline. Verification occurred post-hoc during qualitative error analysis of BERTScore outliers.
* **Code/Result Change:** Manual alteration of rows P019, P060, and P093.
* **Outcome:** Dataset finalized. Manual correction resulted in BERTScore ROC-AUC increasing from 0.9220 to 0.9518.

### Activity 2: Metric Meta-Evaluation Pipeline
* **Task:** Evaluate multiple automated metrics (BERTScore, ROUGE-L, BLEU, Cross-Encoder NLI) against the generated pairs.
* **Tool:** Gemini.
* **AI Proposal:** Provided Python pipeline using `huggingface/transformers`, `bert_score`, and `nltk`. Proposed mathematically inverting similarity metrics (e.g., `-df["score_bertscore"]`) before computing ROC-AUC to align with divergence metrics.
* **Human Decision & Verification:** The user executed the code in a deterministic Colab environment (Seed 42). 
* **Code/Result Change:** Successfully resolved a deprecated Hugging Face `evaluate` bug by swapping to native `nltk`/`rouge_score`.
* **Outcome:** Robust, bootstrapped 95% Confidence Intervals and Spearman correlations were calculated without directional metric leakage.

### Activity 3: Pathway C (Targeted Theoretical Modification)
* **Task:** Rescue the collapsed Cross-Encoder $\Delta_{\text{epistemic}}$ metric.
* **Tool:** Gemini & Qwen2.5-1.5B-Instruct.
* **Prompt/Version:** Inject explicit taxonomic rules into the NLI prompt ("Rule 1: Correlational does NOT entail causal...").
* **AI Proposal:** Drop-in replacement for the cross-encoder using quantized generative text-to-float probability scoring.
* **Human Decision & Verification:** The user approved Pathway C. Verification via CLI output exposed a `KeyError` due to mismatched dataframe column names.
* **Code/Result Change:** Re-aligned the ROC plotting code to target `score_delta_epistemic_llm`.
* **Outcome:** ROC-AUC improved from 0.3234 (cross-encoder) to 0.6002 (Generative LLM).

---

## 3. Real Examples of AI Limitations and Manual Corrections

**Limitation 1: Generative AI Dataset Duplication (ChatGPT)**
Despite strict prompt constraints to "upgrade the correlational verb into a DIRECT CAUSAL verb," ChatGPT failed on several dense sentences. For pairs P019, P060, and P093, it simply outputted the exact original string as the "Escalated" string. 
* *Correction:* The human researcher manually intervened in the CSV, rewriting the verbs (e.g., changing "factors linked to" to "factors responsible for"). 

**Limitation 2: NLI Taxonomic Hallucination and Context Overload (Qwen-1.5B)**
During Pathway C, the explicit rule-injected Generative LLM failed to match BERTScore. Qualitative error extraction revealed two specific AI limitations:
* *Context Overload:* In dense biomedical sentences (e.g., P001), the LLM prioritized 95% token overlap over the localized verb shift, outputting `0.0000` divergence (False Negative).
* *Synonym Hallucination:* When given safe paraphrases (e.g., "associated with" $\rightarrow$ "correlated with" in P050), the LLM rigidly misapplied the prompt rules, hallucinating a massive epistemic shift ($\Delta = 0.9000$).

**Limitation 3: Code Context Blindness in Drop-in Replacements (Gemini)**
When replacing the Cycle 4 (cross-encoder) code with Pathway C (LLM), the AI provided the new metric generation code but failed to simultaneously update the downstream sanity check `print` statement, which still requested the deleted `score_delta_epistemic` column. 
* *Correction:* The user flagged the `['score_delta_epistemic'] not in index` error. The AI subsequently provided a patched visualization cell using the new `score_delta_epistemic_llm` column.

---

## 4. Human Reasoning Dependencies vs. AI Assistance

**Dependent on Human Reasoning:**
* **Methodological Alignment:** The AI assistant initially pointed out that the user's continuous pairwise divergence metric ($\Delta_{\text{epistemic}}$) was fundamentally incompatible with the discrete multi-class sentence classification methodology of the reference paper (Kim et al., 2023). The human decided to pivot the manuscript's framing to a comparative audit of evaluation metrics rather than a direct replication of Kim et al.
* **Data Auditing:** The AI extracted the textual errors, but the human recognized the implications (that identical strings were artificially suppressing BERTScore's ground truth performance) and made the manual syntactic corrections to the dataset.

**Dependent on AI Assistance:**
* **"Vibe Coding" & Boilerplate:** The AI handled the heavy lifting for PyTorch VRAM management, regex parsing of LLM outputs, ROC-AUC bootstrapping calculations, and matplotlib generation. 
* **Manuscript Structuring:** The AI translated raw CSV outputs and debugging realizations (like the MNLI domain bias) into formal academic prose for the LaTeX draft.

## 5. Remaining Risks
* **Evaluator-Generator Contamination:** Because ChatGPT generated the underlying dataset, evaluating it with other LLMs or LLM-trained embeddings (BERTScore) carries a risk of evaluating stylistic alignment rather than genuine epistemic shifts.
* **Metric Extrapolation:** While BERTScore proved near-perfect for single-verb replacements in a fixed context, its reliability in auditing completely restructured, multi-clause edits remains untested.