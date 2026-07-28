import os
import sys


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.exp1_membership_inference.ibd_attack import (
    load_phased_genotypes,
    simulate_child_genotype,
)

SNP_SWEEP = [5, 10, 20, 50, 60, 70, 80, 90, 100, 110, 200]
N_PERMUTATIONS = 1000
N_BOOTSTRAP = 1000
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "exp1")


def _extract_cohort_ids(ped_path: str, sample_ids: list, seed: int = 42):
    ped = pd.read_csv(ped_path, sep="\t")
    for col in ped.columns:
        ped[col] = ped[col].astype(str).str.strip()

    sample_set = set(sample_ids)
    id_to_order = {sid: i for i, sid in enumerate(sample_ids)}  # preserve order

    fathers = ped[
        ped["Relationship"].str.contains("father", case=False, na=False)
        & ped["Individual ID"].isin(sample_set)
    ]
    mothers = ped[
        ped["Relationship"].str.contains("mother", case=False, na=False)
        & ped["Individual ID"].isin(sample_set)
    ]

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
    non_trio_ids = [sid for sid in sample_ids if sid not in trio_members]

    rng = np.random.default_rng(seed)
    shuffled = non_trio_ids.copy()
    rng.shuffle(shuffled)
    current_ctrl_ids = shuffled[: len(mother_ids)]

    return mother_ids, father_ids, non_trio_ids, current_ctrl_ids

def _print_pop_table(mother_ids, ctrl_ids, panel, label="current"):
    m_sub = panel[panel["sample"].isin(mother_ids)]
    c_sub = panel[panel["sample"].isin(ctrl_ids)]

    m_pop  = m_sub["pop"].value_counts()
    c_pop  = c_sub["pop"].value_counts()
    m_spop = m_sub["super_pop"].value_counts()
    c_spop = c_sub["super_pop"].value_counts()

    all_pops = sorted(set(m_pop.index) | set(c_pop.index))
    print(f"\n{'Pop':>6}  {'Mothers N':>9}  {'Mothers %':>9}  "
          f"{'Controls N':>10} ({label})  {'Controls %':>10}")
    print("-" * 65)
    for p in all_pops:
        mn = m_pop.get(p, 0);   mp = mn / len(mother_ids) * 100
        cn = c_pop.get(p, 0);   cp = cn / len(ctrl_ids) * 100
        flag = "  <<" if abs(mp - cp) > 5 else ""
        print(f"{p:>6}  {mn:>9}  {mp:>8.1f}%  {cn:>10}  {cp:>9.1f}%{flag}")

    all_spops = sorted(set(m_spop.index) | set(c_spop.index))
    print(f"\n{'Spop':>6}  {'Mothers N':>9}  {'Mothers %':>9}  "
          f"{'Controls N':>10} ({label})  {'Controls %':>10}")
    print("-" * 65)
    for sp in all_spops:
        mn = m_spop.get(sp, 0);  mp = mn / len(mother_ids) * 100
        cn = c_spop.get(sp, 0);  cp = cn / len(ctrl_ids) * 100
        flag = "  <<" if abs(mp - cp) > 5 else ""
        print(f"{sp:>6}  {mn:>9}  {mp:>8.1f}%  {cn:>10}  {cp:>9.1f}%{flag}")

    return m_pop, c_pop, all_pops


def task1(tsv_path, ped_path, panel_path, seed=42):
    print("\n" + "=" * 70)
    print("TASK 1 — Population Stratification Check")
    print("=" * 70)

    # Load sample IDs from the first row of the genotype TSV (fast — 1 row)
    id_row = pd.read_csv(tsv_path, sep="\t", nrows=1, header=None)
    sample_ids = id_row.iloc[0].tolist()
    print(f"Genotype matrix: {len(sample_ids)} samples")

    panel = pd.read_csv(panel_path, sep="\t", usecols=["sample", "pop", "super_pop"])
    print(f"Panel file: {len(panel)} entries, "
          f"{panel['pop'].nunique()} populations, "
          f"{panel['super_pop'].nunique()} superpopulations")

    mother_ids, father_ids, non_trio_ids, ctrl_ids = _extract_cohort_ids(
        ped_path, sample_ids, seed
    )
    print(f"\nTrio mothers (beacon members) : {len(mother_ids)}")
    print(f"Trio fathers (excluded)        : {len(father_ids)}")
    print(f"Non-trio pool size             : {len(non_trio_ids)}")
    print(f"Current controls (unmatched)   : {len(ctrl_ids)}")

    geno_set = set(sample_ids)
    panel_set = set(panel["sample"])
    uncovered = geno_set - panel_set
    if uncovered:
        print(f"\nWARNING: {len(uncovered)} genotyped samples not in panel file "
              f"— these will be excluded from population tables.")

    print("\n=== (a) Fine-grained population distribution ===")
    m_pop, c_pop, all_pops = _print_pop_table(mother_ids, ctrl_ids, panel, label="unmatched")
    contingency = pd.DataFrame(
        {"Mothers": [m_pop.get(p, 0) for p in all_pops],
         "Controls": [c_pop.get(p, 0) for p in all_pops]},
        index=all_pops,
    )
    contingency = contingency[(contingency > 0).any(axis=1)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chi2, p_val, dof, expected = chi2_contingency(contingency.values)

    print(f"\n=== Chi-square test (fine-grained pop vs. trio status) ===")
    print(f"  Chi-square statistic : {chi2:.4f}")
    print(f"  Degrees of freedom   : {dof}")
    print(f"  p-value              : {p_val:.6f}")

    deviations = []
    for p in all_pops:
        mp = m_pop.get(p, 0) / len(mother_ids) * 100
        cp = c_pop.get(p, 0) / len(ctrl_ids) * 100
        deviations.append((p, abs(mp - cp), mp, cp))
    deviations.sort(key=lambda x: -x[1])
    print(f"\n  Largest per-population deviations (mothers % vs controls %):")
    for pop, dev, mp, cp in deviations[:5]:
        print(f"    {pop:>6}: mothers {mp:.1f}%  controls {cp:.1f}%  |diff|={dev:.1f}pp")

    print()
    if p_val > 0.05:
        print("CONCLUSION (Task 1): p > 0.05 — distributions are NOT significantly")
        print("different. The current control set is adequately matched by population.")
        print("Population stratification is NOT a meaningful confound in this design.")
        print("Tasks 2–4 are NOT required.")
    else:
        print(f"CONCLUSION (Task 1): p = {p_val:.6f} <= 0.05 -- significant population")
        print("mismatch between trio mothers and current controls. Proceeding to Task 2.")

    return (mother_ids, father_ids, non_trio_ids, ctrl_ids,
            panel, p_val, sample_ids)



def task2(mother_ids, non_trio_ids, panel, seed=42):
    print("\n" + "=" * 70)
    print("TASK 2 — Population-Matched Control Sampling")
    print("=" * 70)

    m_sub = panel[panel["sample"].isin(mother_ids)]
    target_counts = m_sub["pop"].value_counts().sort_index()

    # Pool = non-trio individuals present in the panel
    pool_df = panel[panel["sample"].isin(non_trio_ids)].copy()

    rng = np.random.default_rng(seed)
    matched = []
    used = set()

    print(f"\n{'Pop':>6}  {'Target':>7}  {'Pool N':>7}  {'Sampled':>8}  {'Gap':>6}")
    print("-" * 42)

    for pop, target_n in target_counts.items():
        available = pool_df[
            (pool_df["pop"] == pop) & (~pool_df["sample"].isin(used))
        ]["sample"].tolist()
        n_avail = len(available)
        n_sample = min(int(target_n), n_avail)
        sampled = rng.choice(available, size=n_sample, replace=False).tolist()
        matched.extend(sampled)
        used.update(sampled)
        gap = n_sample - int(target_n)
        gap_str = f"{gap:+d}" if gap != 0 else "  0"
        print(f"{pop:>6}  {int(target_n):>7}  {n_avail:>7}  {n_sample:>8}  {gap_str:>6}")

    print(f"\nTotal matched controls: {len(matched)}  (target: {len(mother_ids)})")
    achieved = panel[panel["sample"].isin(matched)]["pop"].value_counts()
    all_pops = sorted(set(target_counts.index) | set(achieved.index))
    print("\n--- Achieved vs. Target (fine-grained pop) ---")
    print(f"{'Pop':>6}  {'Target':>7}  {'Achieved':>9}  {'Match':>6}")
    print("-" * 32)
    all_match = True
    for p in all_pops:
        t = int(target_counts.get(p, 0))
        a = int(achieved.get(p, 0))
        ok = "OK" if a == t else f"OFF({a-t:+d})"
        if a != t:
            all_match = False
        print(f"{p:>6}  {t:>7}  {a:>9}  {ok:>6}")
    if all_match:
        print("\nAll strata exactly matched.")
    else:
        print("\nNote: some strata could not be exactly matched (pool too small).")

    return matched


def _permutation_test(X, y, clf, n_perm, seed):
    rng = np.random.default_rng(seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    probs_obs = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    observed = roc_auc_score(y, probs_obs)
    null_aucs = []
    y_arr = np.array(y)
    for _ in range(n_perm):
        yp = rng.permutation(y_arr)
        pp = cross_val_predict(clf, X, yp, cv=cv, method="predict_proba")[:, 1]
        null_aucs.append(roc_auc_score(yp, pp))
    p_val = float(np.mean(np.array(null_aucs) >= observed))
    return observed, p_val


def _bootstrap_ci(X, y, clf, n_boot, seed, alpha=0.05):
    rng = np.random.default_rng(seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    probs = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, probs)
    y_arr = np.array(y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_arr), size=len(y_arr))
        if len(np.unique(y_arr[idx])) < 2:
            continue
        boots.append(roc_auc_score(y_arr[idx], probs[idx]))
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return auc, lo, hi


def task3(dosage, haplotypes, sample_ids,
          mother_ids, father_ids, matched_ctrl_ids,
          seed=42):
    print("\n" + "=" * 70)
    print("TASK 3 — Attack rerun with population-matched controls")
    print(f"         ({N_PERMUTATIONS} permutations + {N_BOOTSTRAP} bootstrap per k — this takes a while)")
    print("=" * 70)

    id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
    mother_idx  = [id_to_idx[m] for m in mother_ids  if m in id_to_idx]
    father_idx  = [id_to_idx[f] for f in father_ids  if f in id_to_idx]
    ctrl_idx    = [id_to_idx[c] for c in matched_ctrl_ids if c in id_to_idx]


    valid_pairs = [(mi, fi) for mi, fi in zip(mother_idx, father_idx)
                   if mi < dosage.shape[0] and fi < dosage.shape[0]]
    mother_idx = [p[0] for p in valid_pairs]
    father_idx = [p[1] for p in valid_pairs]

    print(f"Mothers in genotype matrix : {len(mother_idx)}")
    print(f"Matched controls           : {len(ctrl_idx)}")

 
    rng = np.random.default_rng(seed)
    print("Simulating child genomes...")
    simulated_children = np.array([
        simulate_child_genotype(haplotypes[father_idx[i]], haplotypes[mother_idx[i]], rng)
        for i in tqdm(range(len(mother_idx)), desc="Simulating children")
    ])

    ctrl_genotypes = dosage[ctrl_idx]
    X_all = np.vstack([simulated_children, ctrl_genotypes])
    y = np.array([1] * len(mother_idx) + [0] * len(ctrl_idx))

    clf = LogisticRegression(max_iter=1000, C=10)
    results = []

    for k in SNP_SWEEP:
        X = X_all[:, :k]
        print(f"\nk={k}: permutation test...", flush=True)
        obs_auc, p_val = _permutation_test(X, y, clf, N_PERMUTATIONS, seed)
        print(f"k={k}: bootstrap CI...", flush=True)
        _, ci_lo, ci_hi = _bootstrap_ci(X, y, clf, N_BOOTSTRAP, seed)
        sig = "YES" if p_val < 0.05 else "NO"
        print(f"  AUC={obs_auc:.4f}  95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]  "
              f"p={p_val:.4f}  sig={sig}")
        results.append({
            "k (SNPs)": k,
            "Observed AUC": round(obs_auc, 4),
            "95% CI Lower": round(ci_lo, 4),
            "95% CI Upper": round(ci_hi, 4),
            "p-value (permutation)": round(p_val, 4),
            "Significant (p<0.05)": sig,
        })

    df = pd.DataFrame(results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "significance_results_matched_controls.csv")
    df.to_csv(out, index=False)
    print(f"\nMatched-control results saved: {out}")
    return df



def task4(matched_df, original_csv_path):
    print("\n" + "=" * 70)
    print("TASK 4 — Comparison: Unmatched vs. Population-Matched Controls")
    print("=" * 70)

    orig = pd.read_csv(original_csv_path)

    # Align on k
    orig = orig.set_index("k (SNPs)")
    matched = matched_df.set_index("k (SNPs)")

    header = (
        f"\n{'k':>6}  "
        f"{'Orig AUC':>9}  {'Orig CI':>15}  {'Orig p':>7}  {'Orig sig':>8}  |  "
        f"{'Match AUC':>9}  {'Match CI':>15}  {'Match p':>7}  {'Match sig':>9}"
    )
    print(header)
    print("-" * len(header.strip()))

    auc_diffs = []
    for k in SNP_SWEEP:
        o = orig.loc[k]
        m = matched.loc[k]
        o_auc = o["Observed AUC"]
        m_auc = m["Observed AUC"]
        auc_diffs.append(m_auc - o_auc)
        o_ci  = f"[{o['95% CI Lower']:.3f},{o['95% CI Upper']:.3f}]"
        m_ci  = f"[{m['95% CI Lower']:.3f},{m['95% CI Upper']:.3f}]"
        o_sig = o["Significant (p<0.05)"]
        m_sig = m["Significant (p<0.05)"]
        print(
            f"{k:>6}  "
            f"{o_auc:>9.4f}  {o_ci:>15}  {o['p-value (permutation)']:>7.4f}  {o_sig:>8}  |  "
            f"{m_auc:>9.4f}  {m_ci:>15}  {m['p-value (permutation)']:>7.4f}  {m_sig:>9}"
        )

    mean_diff = float(np.mean(auc_diffs))
    max_diff  = float(np.max(np.abs(auc_diffs)))
    orig_onset_k = None
    match_onset_k = None
    for k in SNP_SWEEP:
        if orig.loc[k, "Significant (p<0.05)"] == "YES" and orig_onset_k is None:
            orig_onset_k = k
        if matched.loc[k, "Significant (p<0.05)"] == "YES" and match_onset_k is None:
            match_onset_k = k

    print("\n--- Summary statistics ---")
    print(f"  Mean AUC shift (matched − original): {mean_diff:+.4f}")
    print(f"  Max |AUC| shift across k:            {max_diff:.4f}")
    print(f"  Significance onset (original):       k={orig_onset_k}")
    print(f"  Significance onset (matched):        k={match_onset_k}")

    # Honest interpretation
    print("\n--- INTERPRETATION (Task 4) ---")
    MATERIAL_THRESHOLD = 0.01   # >1 pp mean AUC shift = material
    ONSET_SHIFT = orig_onset_k != match_onset_k

    if max_diff < MATERIAL_THRESHOLD and not ONSET_SHIFT:
        print(
            "The AUC curve is materially UNCHANGED after population matching\n"
            f"(mean shift = {mean_diff:+.4f}, max |shift| = {max_diff:.4f},\n"
            f"significance onset unchanged at k={orig_onset_k}).\n\n"
            "This confirms that the IBD signal is attributable to the first-degree\n"
            "relative relationship, NOT to population stratification between the\n"
            "trio-mother cohort and the control group. The unmatched-control results\n"
            "in the paper are valid and do not need to be replaced."
        )
    else:
        reasons = []
        if max_diff >= MATERIAL_THRESHOLD:
            reasons.append(
                f"max |AUC shift| = {max_diff:.4f} (threshold: {MATERIAL_THRESHOLD:.2f})"
            )
        if ONSET_SHIFT:
            reasons.append(
                f"significance onset shifted from k={orig_onset_k} to k={match_onset_k}"
            )
        print(
            "POPULATION STRUCTURE WAS A REAL CONFOUND.\n"
            f"Reason(s): {'; '.join(reasons)}.\n\n"
            "The matched-control numbers are the corrected result and should replace\n"
            "the original Table I in the paper. The unmatched AUC values must NOT\n"
            "be cited as the primary result. Flag this for revision before submission."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Stratification check for Exp 1 IBD attack (Tasks 1–4)"
    )
    parser.add_argument(
        "--genotype-tsv",
        default="data/processed/genotype_matrix.tsv",
    )
    parser.add_argument(
        "--ped",
        default="data/raw/20130606_g1k.ped",
    )
    parser.add_argument(
        "--panel",
        default="data/raw/integrated_call_samples_v3.20130502.ALL.panel",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force-tasks-234",
        action="store_true",
        help="Run Tasks 2–4 even if chi-square p > 0.05 (for sensitivity analysis)",
    )
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "..", "..")
    tsv   = os.path.join(base, args.genotype_tsv) if not os.path.isabs(args.genotype_tsv) else args.genotype_tsv
    ped   = os.path.join(base, args.ped)           if not os.path.isabs(args.ped)           else args.ped
    panel = os.path.join(base, args.panel)         if not os.path.isabs(args.panel)         else args.panel
    orig_csv = os.path.join(base, "results", "exp1", "significance_results.csv")

    (mother_ids, father_ids, non_trio_ids, orig_ctrl_ids,
     panel_df, p_val, sample_ids) = task1(tsv, ped, panel, args.seed)

    proceed = (p_val <= 0.05) or args.force_tasks_234
    if not proceed:
        print("\nStopping after Task 1 — no confound detected.")
        return

    matched_ctrl_ids = task2(mother_ids, non_trio_ids, panel_df, args.seed)

    print("\nLoading full phased genotype matrix for attack rerun...")
    dosage, haplotypes, geno_sample_ids = load_phased_genotypes(tsv, n_snps=500)

    matched_df = task3(
        dosage, haplotypes, geno_sample_ids,
        mother_ids, father_ids, matched_ctrl_ids,
        seed=args.seed,
    )

    task4(matched_df, orig_csv)


if __name__ == "__main__":
    main()
