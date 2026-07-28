import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.exp1_membership_inference.ibd_attack import load_phased_genotypes, build_ibd_cohort, simulate_child_genotype
from src.exp1_membership_inference.baselines import shringarpure_bustamante_lrt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "exp1")
N_PERMUTATIONS = 1000
N_BOOTSTRAP = 1000
SNP_SWEEP = [5, 10, 20, 50, 60, 70, 80, 90, 100, 110, 200]


def permutation_test(X, y, clf, n_permutations, seed):
    rng = np.random.default_rng(seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


    probs_obs = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    observed_auc = roc_auc_score(y, probs_obs)

    null_aucs = []
    y_arr = np.array(y)
    for _ in range(n_permutations):
        y_perm = rng.permutation(y_arr)
        probs_perm = cross_val_predict(clf, X, y_perm, cv=cv, method="predict_proba")[:, 1]
        null_aucs.append(roc_auc_score(y_perm, probs_perm))

    null_aucs = np.array(null_aucs)
    p_value = float(np.mean(null_aucs >= observed_auc))
    return observed_auc, null_aucs, p_value


def bootstrap_ci(X, y, clf, n_bootstrap, seed, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    probs = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    base_auc = roc_auc_score(y, probs)

    boot_aucs = []
    y_arr = np.array(y)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_arr[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_arr[idx], probs[idx]))

    boot_aucs = np.array(boot_aucs)
    ci_lo = float(np.percentile(boot_aucs, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_aucs, 100 * (1 - alpha / 2)))
    return base_auc, ci_lo, ci_hi, boot_aucs


def run_significance(genotype_tsv, ped_path, seed=42, n_snps=500):
    dosage, haplotypes, sample_ids = load_phased_genotypes(genotype_tsv, n_snps=n_snps)
    mother_idx, father_idx, control_idx = build_ibd_cohort(ped_path, sample_ids, seed)

    rng = np.random.default_rng(seed)
    simulated_children = np.array([
        simulate_child_genotype(haplotypes[father_idx[i]], haplotypes[mother_idx[i]], rng)
        for i in tqdm(range(len(mother_idx)), desc="Simulating children")
    ])

    control_genotypes = dosage[control_idx]
    X_all = np.vstack([simulated_children, control_genotypes])
    y = np.array([1] * len(mother_idx) + [0] * len(control_idx))

    print(f"\nRunning significance tests across k-sweep ({N_PERMUTATIONS} permutations, {N_BOOTSTRAP} bootstrap samples per k)...")
    print("This will take a while — ~10-20 min depending on hardware.\n")

    results = []
    clf = LogisticRegression(max_iter=1000, C=10)

    for k in SNP_SWEEP:
        X = X_all[:, :k]
        print(f"k={k}: permutation test...", flush=True)
        obs_auc, null_dist, p_val = permutation_test(X, y, clf, N_PERMUTATIONS, seed)
        null_mean = float(np.mean(null_dist))
        null_std = float(np.std(null_dist))

        print(f"k={k}: bootstrap CI...", flush=True)
        _, ci_lo, ci_hi, _ = bootstrap_ci(X, y, clf, N_BOOTSTRAP, seed)

        significant = "YES" if p_val < 0.05 else "NO"
        ci_excludes_chance = "YES" if ci_lo > 0.50 else "NO"

        print(f"  AUC={obs_auc:.4f}  95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]  "
              f"p={p_val:.4f}  sig={significant}  CI>0.50={ci_excludes_chance}")

        results.append({
            "k (SNPs)": k,
            "Observed AUC": round(obs_auc, 4),
            "95% CI Lower": round(ci_lo, 4),
            "95% CI Upper": round(ci_hi, 4),
            "Null Mean AUC": round(null_mean, 4),
            "Null Std AUC": round(null_std, 4),
            "p-value (permutation)": round(p_val, 4),
            "Significant (p<0.05)": significant,
            "CI Excludes 0.50": ci_excludes_chance,
        })

    df = pd.DataFrame(results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "significance_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved: {out_path}")
    print("\n" + df.to_string(index=False))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--genotype-tsv", default="data/processed/genotype_matrix.tsv")
    parser.add_argument("--ped", default="data/raw/20130606_g1k.ped")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-snps", type=int, default=500)
    args = parser.parse_args()
    run_significance(args.genotype_tsv, args.ped, args.seed, args.n_snps)
