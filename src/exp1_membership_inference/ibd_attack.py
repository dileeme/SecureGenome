import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.exp1_membership_inference.baselines import compute_baseline_aucs

SNP_SWEEP = [5, 10, 20, 50, 60, 70, 80, 90, 100, 110, 200]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "exp1")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "figures")


def load_phased_genotypes(tsv_path: str, n_snps: int = 500):
    print(f"Loading phased genotype matrix ({n_snps} SNPs)...")
    df = pd.read_csv(tsv_path, sep="\t", nrows=n_snps + 1, header=None)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    sample_ids = df.iloc[0].tolist()
    df = df.iloc[1:].reset_index(drop=True)

    n_actual_snps = len(df)
    n_samples = len(sample_ids)
    dosage = np.zeros((n_samples, n_actual_snps), dtype=np.int8)
    haplotypes = np.zeros((n_samples, n_actual_snps, 2), dtype=np.int8)


    raw = df.values.T

    for j in range(n_actual_snps):
        col = raw[:, j]
        for i, g in enumerate(col):
            g = str(g).replace("/", "|")
            a, b = g.split("|") if "|" in g else (g[0], g[-1])
            a, b = int(a), int(b)
            haplotypes[i, j, 0] = a
            haplotypes[i, j, 1] = b
            dosage[i, j] = a + b

    print(f"Loaded: {n_samples} samples x {n_actual_snps} SNPs.")
    return dosage, haplotypes, sample_ids


def build_ibd_cohort(ped_path: str, sample_ids: list, seed: int = 42):
  
    ped = pd.read_csv(ped_path, sep="\t")
    for col in ped.columns:
        ped[col] = ped[col].astype(str).str.strip()

    sample_set = set(sample_ids)
    id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}


    trios = ped[
        (ped["Paternal ID"] != "0") &
        (ped["Maternal ID"] != "0") &
        ped["Paternal ID"].isin(sample_set) &
        ped["Maternal ID"].isin(sample_set)
    ].copy()

    fathers = ped[ped["Relationship"].str.contains("father", case=False, na=False) &
                  ped["Individual ID"].isin(sample_set)]
    mothers = ped[ped["Relationship"].str.contains("mother", case=False, na=False) &
                  ped["Individual ID"].isin(sample_set)]

    fam_fathers = fathers.set_index("Family ID")["Individual ID"].to_dict()
    fam_mothers = mothers.set_index("Family ID")["Individual ID"].to_dict()

    complete_fams = set(fam_fathers) & set(fam_mothers)

    mother_ids, father_ids = [], []
    for fam in sorted(complete_fams):
        m = fam_mothers[fam]
        f = fam_fathers[fam]
        if m in sample_set and f in sample_set:
            mother_ids.append(m)
            father_ids.append(f)

    trio_members = set(mother_ids) | set(father_ids)
    control_ids = [sid for sid in sample_ids if sid not in trio_members]

    rng = np.random.default_rng(seed)
    rng.shuffle(control_ids)
    
    control_ids = control_ids[:len(mother_ids)]

    mother_indices = [id_to_idx[m] for m in mother_ids]
    father_indices = [id_to_idx[f] for f in father_ids]
    control_indices = [id_to_idx[c] for c in control_ids]

    print(f"Complete trios (both parents genotyped): {len(mother_ids)}")
    print(f"Controls (unrelated, matched n): {len(control_ids)}")
    return mother_indices, father_indices, control_indices




def simulate_child_genotype(
    father_haplotypes: np.ndarray,
    mother_haplotypes: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n_snps = father_haplotypes.shape[0]
    pat_choice = rng.integers(0, 2, size=n_snps)  
    mat_choice = rng.integers(0, 2, size=n_snps) 
    pat_allele = father_haplotypes[np.arange(n_snps), pat_choice]
    mat_allele = mother_haplotypes[np.arange(n_snps), mat_choice]
    return (pat_allele + mat_allele).astype(np.int8)



def run_ibd_attack(genotype_tsv: str, ped_path: str, seed: int = 42, n_snps: int = 500):
    dosage, haplotypes, sample_ids = load_phased_genotypes(genotype_tsv, n_snps=n_snps)
    mother_idx, father_idx, control_idx = build_ibd_cohort(ped_path, sample_ids, seed)

    rng = np.random.default_rng(seed)

    # Simulate one child genome per trio
    print("Simulating child genomes via Mendelian inheritance...")
    simulated_children = np.array([
        simulate_child_genotype(haplotypes[father_idx[i]], haplotypes[mother_idx[i]], rng)
        for i in tqdm(range(len(mother_idx)), desc="Simulating children")
    ]) 


    control_genotypes = dosage[control_idx]  


    X_all = np.vstack([simulated_children, control_genotypes])
    y = np.array([1] * len(mother_idx) + [0] * len(control_idx))

    print(f"\nCohort: {len(mother_idx)} beacon members (mothers), {len(control_idx)} controls")
    print(f"Feature matrix shape: {X_all.shape}")
    print(f"Attack: adversary uses simulated child genome (50% IBD with mother) vs. "
          f"unrelated genome (0% IBD) to infer beacon membership")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    results = []
    fig, ax = plt.subplots(figsize=(10, 6))

    for k in tqdm(SNP_SWEEP, desc="SNP sweep"):
        X = X_all[:, :k]
        maf = dosage[:, :k].mean(axis=0) / 2.0 

        clf = LogisticRegression(max_iter=1000, C=10)
        lr_auc = float(np.mean(
            cross_val_score(clf, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
        ))

        baseline = compute_baseline_aucs(X, y, maf)

        results.append({
            "k (SNPs)": k,
            "LR AUC": round(lr_auc, 4),
            "SB-LRT AUC": round(baseline["sb_lrt_auc"], 4),
            "Raisaro AUC": round(baseline["raisaro_auc"], 4),
        })

        clf.fit(X, y)
        probs = clf.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, probs)
        ax.plot(fpr, tpr, label=f"k={k} LR (AUC={lr_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Exp 1 (IBD): Relative-Assisted MIA — Simulated Child as Adversary Reference")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(alpha=0.3)

    fig_path = os.path.join(FIGURES_DIR, "reid_auc_ibd.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {fig_path}")

    results_df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "reidentification_results_ibd.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved: {csv_path}")

    print("\n" + "=" * 62)
    print(f"{'k':>6}  {'LR AUC':>8}  {'SB-LRT':>8}  {'Raisaro':>8}")
    print("-" * 62)
    for r in results:
        print(f"{r['k (SNPs)']:>6}  {r['LR AUC']:>8.4f}  "
              f"{r['SB-LRT AUC']:>8.4f}  {r['Raisaro AUC']:>8.4f}")
    print("=" * 62)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 1 (IBD): Relative-Assisted Membership Inference Attack"
    )
    parser.add_argument("--genotype-tsv", default="data/processed/genotype_matrix.tsv")
    parser.add_argument("--ped", default="data/raw/20130606_g1k.ped",
                        help="1000G pedigree file (.ped format)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-snps", type=int, default=500,
                        help="Number of SNPs to load from the genotype matrix")
    args = parser.parse_args()

    run_ibd_attack(args.genotype_tsv, args.ped, args.seed, args.n_snps)
