import os
import sys
import pandas as pd
import numpy as np

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "results")
EXP1_CSV = os.path.join(RESULTS_ROOT, "exp1", "reidentification_results.csv")
EXP2_CSV = os.path.join(RESULTS_ROOT, "exp2", "benchmark_results.csv")
EXP3_CSV = os.path.join(RESULTS_ROOT, "exp3", "tuned_benchmark_results.csv")
OUT_CSV = os.path.join(RESULTS_ROOT, "analysis", "cue_results_expanded.csv")


def _get(df: pd.DataFrame, metric: str) -> float:
    matches = df.loc[df["Metric"] == metric, "Value"]
    if matches.empty:
        raise KeyError(f"Metric '{metric}' not found in CSV")
    return float(matches.values[0])


def run_sanity_checks(bench_df: pd.DataFrame, tuned_df: pd.DataFrame):
    errors = []
    n_exp2 = int(_get(bench_df, "N Individuals"))
    n_exp3 = int(_get(tuned_df, "N Individuals"))
    if n_exp2 != n_exp3:
        errors.append(f"Sample size mismatch: Exp2 N={n_exp2}, Exp3 N={n_exp3}")

    enc_time = _get(tuned_df, "Encryption Time (Total) (s)")
    comp_time = _get(tuned_df, "Computation Time (Total) (s)")
    total_computed = enc_time + comp_time
    try:
        total_field = _get(tuned_df, "Total FHE Time (s)")
        if abs(total_field - total_computed) > 0.01:
            errors.append(
                f"Total latency mismatch: enc+comp={total_computed:.2f}s vs field={total_field:.2f}s"
            )
    except KeyError:
        pass

    try:
        pt_ms = _get(bench_df, "Plaintext Time (ms)") / 1000.0
        comp_fhe = _get(bench_df, "Total FHE Computation (s)")
        enc_fhe = _get(bench_df, "Total Encryption Time (s)")
        reported_e2e = _get(bench_df, "Overhead End-to-End (x)")
        computed_e2e = (enc_fhe + comp_fhe) / pt_ms
        if abs(reported_e2e - computed_e2e) > 1.0:
            errors.append(
                f"Overhead inconsistency: computed end-to-end={computed_e2e:.0f}x, "
                f"reported={reported_e2e:.0f}x"
            )
    except KeyError:
        pass

    if errors:
        print("\nSANITY CHECK FAILURES:")
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(1)
    else:
        print("Sanity checks passed.")


def run_cue():
    for path in [EXP1_CSV, EXP2_CSV, EXP3_CSV]:
        if not os.path.exists(path):
            print(f"Missing results file: {path}")
            print("Run all experiment scripts first before running cue.py.")
            sys.exit(1)

    reid_df = pd.read_csv(EXP1_CSV)
    bench_df = pd.read_csv(EXP2_CSV)
    tuned_df = pd.read_csv(EXP3_CSV)

    run_sanity_checks(bench_df, tuned_df)

    orig_enc = _get(bench_df, "Total Encryption Time (s)")
    orig_comp = _get(bench_df, "Total FHE Computation (s)")
    orig_time = orig_enc + orig_comp

    tuned_enc = _get(tuned_df, "Encryption Time (Total) (s)")
    tuned_comp = _get(tuned_df, "Computation Time (Total) (s)")
    tuned_time = tuned_enc + tuned_comp

    utility_mae = _get(bench_df, "Accuracy (MAE)")

    auc_col = "LR AUC" if "LR AUC" in reid_df.columns else "Mean AUC"

    reid_df = reid_df.sort_values("k (SNPs)").reset_index(drop=True)
    reid_df["Risk_Velocity"] = reid_df[auc_col].diff() / reid_df["k (SNPs)"].diff()
    reid_df["Risk_Acceleration"] = reid_df["Risk_Velocity"].diff() / reid_df["k (SNPs)"].diff()

    cue = reid_df.copy()
    cue["Tuned_Latency_Sec"] = tuned_time
    cue["Latency_Per_SNP_ms"] = (tuned_time / cue["k (SNPs)"]) * 1000
    cue["Efficiency_Gain_Pct"] = round(((orig_time - tuned_time) / orig_time) * 100, 4)
    cue["Utility_MAE"] = utility_mae

    conditions = [
        (cue[auc_col] < 0.70),
        (cue[auc_col] >= 0.70) & (cue[auc_col] < 0.90),
        (cue[auc_col] >= 0.90),
    ]
    tiers = ["SECURE", "VULNERABLE", "CRITICAL_RISK"]
    cue["Security_Tier"] = np.select(conditions, tiers, default="UNKNOWN")
    cue = cue.fillna(0)

    final_cols = [
        "k (SNPs)", auc_col, "Security_Tier", "Risk_Velocity",
        "Risk_Acceleration", "Tuned_Latency_Sec", "Latency_Per_SNP_ms",
        "Efficiency_Gain_Pct", "Utility_MAE",
    ]
    if "SB-LRT AUC" in cue.columns:
        final_cols.insert(2, "SB-LRT AUC")
        final_cols.insert(3, "Raisaro AUC")

    cue = cue[[c for c in final_cols if c in cue.columns]]

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    cue.to_csv(OUT_CSV, index=False)
    print(f"CUE table saved: {OUT_CSV}")
    print(cue.to_string(index=False))


if __name__ == "__main__":
    run_cue()
