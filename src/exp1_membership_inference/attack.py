import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.exp1_membership_inference.cohort_construction import construct_label_external
from src.exp1_membership_inference.baselines import compute_baseline_aucs

SNP_SWEEP = [5, 10, 20, 50, 60, 70, 80, 90, 100, 110, 200]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "exp1")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "figures")


def load_genotype_matrix(tsv_path: str, n_snps: int = 500) -> tuple:
    """Load and parse a VCF-derived genotype TSV. Returns (X_all, sample_ids)."""
    print(f"Loading genotype matrix from {tsv_path} (first {n_snps} SNPs)...")
    df = pd.read_csv(tsv_path, sep="\t", header=None, nrows=n_snps)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")


    if df.iloc[0].dtype == object:
        sample_ids = df.iloc[0].tolist()
        df = df.iloc[1:]
    else:
        sample_ids = [str(i) for i in range(df.shape[1])]

    X_all = df.T.replace({
        "0|0": 0, "0|1": 1, "1|0": 1, "1|1": 2,
        "0/0": 0, "0/1": 1, "1/0": 1, "1/1": 2,
    })
    X_all = X_all.apply(pd.to_numeric, errors="coerce").fillna(0).values
    print(f"Matrix ready: {X_all.shape[0]} individuals, {X_all.shape[1]} SNPs.")
    return X_all, sample_ids


def run_attack(genotype_tsv: str, panel_path: str, superpopulation: str = "EUR", seed: int = 42):
    X_all, all_sample_ids = load_genotype_matrix(genotype_tsv)

    member_sample_ids, y_full = construct_label_external(panel_path, superpopulation, seed)

    id_to_idx = {sid: i for i, sid in enumerate(all_sample_ids)}
    matched_indices = [id_to_idx[sid] for sid in member_sample_ids if sid in id_to_idx]

    if len(matched_indices) == len(member_sample_ids):
        X_sub = X_all[matched_indices]
        y = y_full
    else:
        print(
            f"Warning: only {len(matched_indices)}/{len(member_sample_ids)} sample IDs "
            "matched between panel and genotype matrix. Using positional alignment."
        )
        n = min(len(y_full), X_all.shape[0])
        X_sub = X_all[:n]
        y = y_full[:n]

    print(f"Cohort: {X_sub.shape[0]} individuals, {int(y.sum())} members, {int((1-y).sum())} controls.")
    print(f"Label source: {superpopulation} superpopulation split, panel={panel_path}")
    print(f"SNP sweep: {SNP_SWEEP}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    results = []
    fig, ax = plt.subplots(figsize=(10, 6))

    for k in tqdm(SNP_SWEEP, desc="SNP feature sweep"):
        X = X_sub[:, :k]
        maf = X.mean(axis=0) / 2.0 


        clf = LogisticRegression(max_iter=1000, C=10)
        lr_auc = float(np.mean(cross_val_score(clf, X, y, cv=5, scoring="roc_auc", n_jobs=-1)))
        baseline_aucs = compute_baseline_aucs(X, y, maf)

        results.append({
            "k (SNPs)": k,
            "LR AUC": round(lr_auc, 4),
            "SB-LRT AUC": round(baseline_aucs["sb_lrt_auc"], 4),
            "Raisaro AUC": round(baseline_aucs["raisaro_auc"], 4),
        })

        clf.fit(X, y)
        probs = clf.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, probs)
        ax.plot(fpr, tpr, label=f"k={k} LR (AUC={lr_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Experiment 1: Membership Inference Attack — External Cohort Labels")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(alpha=0.3)

    fig_path = os.path.join(FIGURES_DIR, "reid_auc_curves.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {fig_path}")

    results_df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "reidentification_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved: {csv_path}")

    print("\n" + "=" * 60)
    print(f"{'k':>6}  {'LR AUC':>8}  {'SB-LRT':>8}  {'Raisaro':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['k (SNPs)']:>6}  {r['LR AUC']:>8.4f}  {r['SB-LRT AUC']:>8.4f}  {r['Raisaro AUC']:>8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 1: Membership Inference Attack")
    parser.add_argument("--genotype-tsv", default="data/processed/genotype_matrix.tsv",
                        help="Path to genotype matrix TSV")
    parser.add_argument("--panel",
                        default="data/raw/integrated_call_samples_v3.20130502.ALL.panel",
                        help="Path to 1000G panel file (sample -> superpopulation mapping)")
    parser.add_argument("--superpopulation", default="EUR",
                        help="Superpopulation for cohort construction (default: EUR)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_attack(args.genotype_tsv, args.panel, args.superpopulation, args.seed)
