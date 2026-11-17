import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def get_bootstrap_ci(y_t, y_s, n_bootstraps=2000, seed=42):
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_s), len(y_s))
        if len(np.unique(y_t[indices])) < 2: continue
        score = roc_auc_score(y_t[indices], y_s[indices])
        bootstrapped_scores.append(score)
    return np.percentile(bootstrapped_scores, 2.5), np.percentile(bootstrapped_scores, 97.5)


def compute_meta_evaluation_and_plot(df, run_name="baseline"):
    print(f"\n--- Running Meta-Evaluation ({run_name}) ---")
    os.makedirs("results/Figures", exist_ok=True)

    y_true = df["ground_truth_label"].values

    metrics = {
        "BERTScore": -df["score_bertscore"].values,
        "ROUGE-L": -df["score_rougeL"].values,
        "BLEU": -df["score_bleu"].values
    }
    
    # Dynamically select which NLI metric is present
    if "score_delta_epistemic_llm" in df.columns:
        metrics["Δ_epistemic (LLM)"] = df["score_delta_epistemic_llm"].values
    if "score_delta_epistemic" in df.columns:
        metrics["Δ_epistemic (Cross-Encoder)"] = df["score_delta_epistemic"].values

    print("\n" + "="*70)
    print(f"{'Metric':<25} | {'ROC-AUC':<10} | {'95% CI':<20} | {'Spearman r'}")
    print("-" * 70)

    for name, scores in metrics.items():
        auc = roc_auc_score(y_true, scores)
        ci_lower, ci_upper = get_bootstrap_ci(y_true, scores)

        raw_scores = scores if "Δ_epistemic" in name else -scores
        rho, _ = spearmanr(raw_scores, y_true)

        print(f"{name:<25} | {auc:<10.4f} | [{ci_lower:.3f}, {ci_upper:.3f}]       | {rho:.4f}")
    print("="*70)

    # 1. Figure 1: ROC Curves
    plt.figure(figsize=(8, 6), dpi=300)
    for name, scores in metrics.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves: Epistemic Shift Detection ({run_name})")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(f"results/Figures/ROC_Curve_{run_name}.png", bbox_inches='tight')
    plt.close()

    # 2. Figure 2: KDE Density Distribution
    plt.figure(figsize=(8, 5), dpi=300)
    target_metric = "score_delta_epistemic_llm" if "score_delta_epistemic_llm" in df.columns else "score_delta_epistemic"
    
    sns.kdeplot(df[df["ground_truth_label"] == 0][target_metric], label="Paraphrase (Control)", fill=True, color="blue")
    sns.kdeplot(df[df["ground_truth_label"] == 1][target_metric], label="Escalated (Test)", fill=True, color="red")
    
    plt.xlabel("Δ_epistemic Score")
    plt.ylabel("Density")
    plt.title(f"Distribution of Epistemic Shifts ({run_name})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"results/Figures/KDE_Distribution_{run_name}.png", bbox_inches='tight')
    plt.close()

    output_path = f"results/esm_evaluation_results_{run_name}.csv"
    df.to_csv(output_path, index=False)
    print(f"\nVisualizations saved to 'results/Figures/'. Data saved to '{output_path}'.")


def extract_error_analysis(results_df, raw_dataset_path):
    print("\nLoading data for qualitative error analysis...")
    try:
        text_df = pd.read_csv(raw_dataset_path)
    except FileNotFoundError as e:
        print(f"Error loading raw dataset: {e}")
        return

    orig_texts = text_df[text_df["condition"] == "original"].set_index("pair_id")["text"].to_dict()

    def get_hypothesis(row):
        matches = text_df[(text_df["pair_id"] == row["pair_id"]) & (text_df["condition"] == row["test_condition"])]["text"].values
        return matches[0] if len(matches) > 0 else "TEXT NOT FOUND"

    results_df["reference_text"] = results_df["pair_id"].map(orig_texts)
    results_df["hypothesis_text"] = results_df.apply(get_hypothesis, axis=1)

    print("\n" + "="*80)
    print(" BERTScore ALGORITHMIC BLIND SPOTS (Top 3 Errors)")
    print("="*80)
    print("\n--- FALSE NEGATIVES (Escalations missed by BERTScore) ---")
    fn_bert = results_df[results_df["ground_truth_label"] == 1].sort_values("score_bertscore", ascending=False).head(3)
    for _, row in fn_bert.iterrows():
        print(f"\n[Pair: {row['pair_id']} | BERTScore: {row['score_bertscore']:.4f}]")
        print(f"Original : {row['reference_text']}")
        print(f"Escalated: {row['hypothesis_text']}")

    target_metric = "score_delta_epistemic_llm" if "score_delta_epistemic_llm" in results_df.columns else "score_delta_epistemic"
    
    print("\n" + "="*80)
    print(f" Δ_EPISTEMIC LOGIC COLLAPSE (Top 3 False Positives)")
    print("="*80)
    print("\n--- FALSE POSITIVES (Safe paraphrases flagged as escalations) ---")
    fp_delta = results_df[results_df["ground_truth_label"] == 0].sort_values(target_metric, ascending=False).head(3)
    for _, row in fp_delta.iterrows():
        print(f"\n[Pair: {row['pair_id']} | Delta: {row[target_metric]:.4f}]")
        print(f"Original : {row['reference_text']}")
        print(f"Paraphrase: {row['hypothesis_text']}")
