import pandas as pd
import os

def load_and_prepare_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    # 1. Schema Validation
    required_cols = {'pair_id', 'condition', 'text'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns. Expected {required_cols}, found {set(df.columns)}")

    processed_rows = []
    skipped_pairs = 0

    # 2. Grouping and Pair Generation
    for pair_id, group in df.groupby("pair_id"):
        # Ensure exact triplet structure
        if len(group) != 3:
            skipped_pairs += 1
            continue

        orig = group[group["condition"] == "original"]["text"].values
        para = group[group["condition"] == "paraphrase"]["text"].values
        esc = group[group["condition"] == "escalated"]["text"].values

        # Ensure exactly one of each condition exists
        if len(orig) != 1 or len(para) != 1 or len(esc) != 1:
            skipped_pairs += 1
            continue

        orig_text, para_text, esc_text = orig[0], para[0], esc[0]

        # Pair 1: Negative Control (Paraphrase)
        processed_rows.append({
            "pair_id": pair_id,
            "test_condition": "paraphrase",
            "reference_text": orig_text,
            "hypothesis_text": para_text,
            "ground_truth_label": 0
        })

        # Pair 2: Test Case (Escalation)
        processed_rows.append({
            "pair_id": pair_id,
            "test_condition": "escalated",
            "reference_text": orig_text,
            "hypothesis_text": esc_text,
            "ground_truth_label": 1
        })

    processed_df = pd.DataFrame(processed_rows)
    print(f"Successfully processed {len(processed_df) // 2} complete triplets.")
    if skipped_pairs > 0:
        print(f"WARNING: Skipped {skipped_pairs} pairs due to malformed conditions.")

    return processed_df

# --- Sanity/Unit Test ---
if __name__ == "__main__":
    file_name = "/content/drive/MyDrive/ATR/ESM_Research_Project/data/custom_epistemic_suite.csv"

    # Optional: Create a dummy file if the real one isn't in the current dir for the test to run
    if not os.path.exists(file_name):
        print(f"Dummy creation: {file_name} not found, generating a synthetic one for the test.")
        pd.DataFrame([
            {"pair_id": "P001", "condition": "original", "text": "A is associated with B."},
            {"pair_id": "P001", "condition": "paraphrase", "text": "A is correlated with B."},
            {"pair_id": "P001", "condition": "escalated", "text": "A causes B."}
        ]).to_csv(file_name, index=False)

    try:
        eval_df = load_and_prepare_data(file_name)
        print("\n--- Output Shapes and Types ---")
        print(f"DataFrame Shape: {eval_df.shape} (Expected: N*2 rows, 5 columns)")
        print("\n--- First 2 Rows Preview ---")
        print(eval_df.head(2).to_string())
    except Exception as e:
        print(f"TEST FAILED: {str(e)}")
