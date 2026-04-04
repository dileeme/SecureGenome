"""
Experiment 1: Privacy-Preserving Polygenic Risk Score Computation via BFV-FHE
Dataset:      1000 Genomes Project Phase 3, chr22 VCF (local)
PRS Weights:  PGS Catalog PGS000018 (Type 2 Diabetes) — local
Library:      OpenFHE (BFV scheme)

Measures:
  - Latency (ms) : plaintext vs FHE PRS computation
  - Memory (MB)  : peak RSS delta under both modes
  - Overhead multiplier across SNP counts: 100, 250, 500, 1000

Output: exp1_results.csv, exp1_overhead_plot.png
"""

import os, sys, gzip, time, tracemalloc, io, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import openfhe

# ─── CONFIG ──────────────────────────────────────────────────────────────────

DATA_DIR    = "./data"
VCF_GZ      = os.path.join(DATA_DIR, "ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz")
PGS_FILE    = os.path.join(DATA_DIR, "PGS000018.txt.gz")
RESULTS_CSV = "exp1_results.csv"
PLOT_FILE   = "exp1_overhead_plot.png"

SNP_COUNTS = [100, 250, 500, 1000]
N_SAMPLES  = 50
SEED       = 42

# ─── SANITY CHECK ────────────────────────────────────────────────────────────

def check_files():
    missing = [f for f in [VCF_GZ, PGS_FILE] if not os.path.exists(f)]
    if missing:
        print("[error] Missing files:")
        for f in missing:
            print(f"  {f}")
        print("\nFiles found in ./data/:")
        for f in os.listdir(DATA_DIR):
            print(f"  {f}")
        sys.exit(1)
    print(f"  [ok] VCF:      {VCF_GZ}")
    print(f"  [ok] PGS file: {PGS_FILE}")

# ─── PGS WEIGHTS ─────────────────────────────────────────────────────────────

def fetch_pgs_weights(path):
    with gzip.open(path, "rt") as f:
        lines = [l for l in f if not l.startswith("#")]
    df = pd.read_csv(io.StringIO("".join(lines)), sep="\t", low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={
        "chr_name":      "chrom",
        "chr_position":  "pos",
        "effect_allele": "ea",
        "effect_weight": "weight"
    })
    df["chrom"]  = df["chrom"].astype(str).str.replace("^chr", "", regex=True)
    df["pos"]    = pd.to_numeric(df["pos"],    errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    return df.dropna(subset=["chrom", "pos", "weight"])

# ─── GENOTYPE EXTRACTION ─────────────────────────────────────────────────────

def extract_genotypes(vcf_gz, chrom, positions_set, n_samples=50):
    from cyvcf2 import VCF
    matched_geno = []
    matched_pos  = []
    vcf = VCF(vcf_gz)
    sample_idx = list(range(min(n_samples, len(vcf.samples))))
    print(f"  [info] VCF has {len(vcf.samples)} samples, using first {len(sample_idx)}")
    for variant in vcf(chrom):
        if variant.POS in positions_set:
            gt = variant.gt_types
            dosage = np.where(gt == 3, np.nan, gt.astype(float))
            matched_geno.append(dosage[sample_idx])
            matched_pos.append(variant.POS)
            if len(matched_pos) >= max(SNP_COUNTS):
                break
    vcf.close()
    return np.array(matched_geno), matched_pos  # (n_snps, n_samples)

# ─── PLAINTEXT PRS ───────────────────────────────────────────────────────────

def prs_plaintext(geno_matrix, weights):
    col_means = np.nanmean(geno_matrix, axis=1, keepdims=True)
    filled = np.where(np.isnan(geno_matrix), col_means, geno_matrix)
    return weights @ filled  # (n_samples,)

# ─── BFV-FHE PRS ─────────────────────────────────────────────────────────────

def prs_fhe(geno_matrix, weights):
    n_snps, n_samples = geno_matrix.shape

    # batch size MUST be a power of 2 for BFV
    batch = 2 ** int(math.floor(math.log2(min(n_snps, 4096))))

    col_means = np.nanmean(geno_matrix, axis=1, keepdims=True)
    filled = np.where(np.isnan(geno_matrix), col_means, geno_matrix)

    SCALE = 100
    weights_int = [int(round(w * SCALE)) for w in weights[:batch]]
    weights_int += [0] * (batch - len(weights_int))

    params = openfhe.CCParamsBFVRNS()
    params.SetPlaintextModulus(786433)
    params.SetMultiplicativeDepth(1)
    params.SetBatchSize(batch)
    params.SetSecurityLevel(openfhe.SecurityLevel.HEStd_128_classic)

    cc = openfhe.GenCryptoContext(params)
    cc.Enable(openfhe.PKESchemeFeature.PKE)
    cc.Enable(openfhe.PKESchemeFeature.KEYSWITCH)
    cc.Enable(openfhe.PKESchemeFeature.LEVELEDSHE)

    keypair = cc.KeyGen()
    cc.EvalSumKeyGen(keypair.secretKey)
    cc.EvalMultKeyGen(keypair.secretKey)

    pt_weights = cc.MakePackedPlaintext(weights_int)
    prs_values = []

    for s in range(n_samples):
        geno_int = [int(round(filled[i, s] * SCALE)) for i in range(batch)]
        pt_geno  = cc.MakePackedPlaintext(geno_int)
        ct_geno  = cc.Encrypt(keypair.publicKey, pt_geno)
        ct_prod  = cc.EvalMult(ct_geno, pt_weights)
        ct_sum   = cc.EvalSum(ct_prod, batch)
        pt_res   = cc.Decrypt(ct_sum, keypair.secretKey)
        pt_res.SetLength(1)
        prs_values.append(pt_res.GetPackedValue()[0] / (SCALE * SCALE))

    return prs_values

# ─── BENCHMARK ───────────────────────────────────────────────────────────────

def benchmark(geno_matrix, weights, n_snps):
    G = geno_matrix[:n_snps, :]
    W = weights[:n_snps]

    tracemalloc.start()
    t0 = time.perf_counter()
    prs_plain = prs_plaintext(G, W)
    t_plain = (time.perf_counter() - t0) * 1000
    _, plain_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    t0 = time.perf_counter()
    prs_enc = prs_fhe(G, W)
    t_fhe = (time.perf_counter() - t0) * 1000
    _, fhe_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    corr = np.corrcoef(prs_plain, prs_enc)[0, 1]

    return {
        "n_snps":           n_snps,
        "plaintext_ms":     round(t_plain, 2),
        "fhe_ms":           round(t_fhe, 2),
        "overhead_x":       round(t_fhe / t_plain, 1),
        "plaintext_mem_mb": round(plain_peak / 1e6, 3),
        "fhe_mem_mb":       round(fhe_peak / 1e6, 3),
        "mem_overhead_x":   round(fhe_peak / max(plain_peak, 1), 1),
        "prs_correlation":  round(corr, 4),
    }

# ─── PLOT ────────────────────────────────────────────────────────────────────

def plot_results(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        "FHE vs Plaintext PRS — Compute & Memory Overhead (BFV, 128-bit security)\n"
        "Dataset: 1000 Genomes Phase 3 chr22 | Weights: PGS000018 (T2D)",
        fontsize=10, fontweight="bold"
    )

    ax = axes[0]
    ax.plot(df.n_snps, df.plaintext_ms, "o-", label="Plaintext", color="#2ecc71")
    ax.plot(df.n_snps, df.fhe_ms,       "s-", label="BFV-FHE",   color="#e74c3c")
    ax.set_xlabel("SNP Count"); ax.set_ylabel("Time (ms)")
    ax.set_title("Latency"); ax.legend(); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    ax = axes[1]
    ax.bar(df.n_snps.astype(str), df.overhead_x, color="#3498db", edgecolor="white")
    ax.set_xlabel("SNP Count"); ax.set_ylabel("Overhead (×)")
    ax.set_title("Compute Overhead Multiplier")
    for i, v in enumerate(df.overhead_x):
        ax.text(i, v + 0.5, f"{v}×", ha="center", fontsize=9)

    ax = axes[2]
    ax.plot(df.n_snps, df.plaintext_mem_mb, "o-", label="Plaintext", color="#2ecc71")
    ax.plot(df.n_snps, df.fhe_mem_mb,       "s-", label="BFV-FHE",   color="#e74c3c")
    ax.set_xlabel("SNP Count"); ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Memory Footprint"); ax.legend()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    print(f"  [saved] {PLOT_FILE}")

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Experiment 1: FHE PRS Benchmark ===\n")
    check_files()

    # 1. Load PGS weights
    print("\n  [load] Parsing PGS000018 weights...")
    pgs = fetch_pgs_weights(PGS_FILE)
    chr22_pgs = pgs[pgs["chrom"] == "22"].reset_index(drop=True)
    print(f"  [info] PGS000018 variants on chr22: {len(chr22_pgs)}")

    if len(chr22_pgs) >= 50:
        pgs_use = chr22_pgs.head(max(SNP_COUNTS)).copy()
    else:
        print("  [warn] Few chr22 PGS SNPs — using genome-wide top SNPs")
        pgs_use = pgs.head(max(SNP_COUNTS)).copy()

    target_positions = set(pgs_use["pos"].dropna().astype(int).tolist())

    # 2. Extract genotypes — try "22" then "chr22"
    print(f"\n  [extract] Pulling matched SNPs from VCF (may take a few minutes)...")
    geno_matrix, matched_pos = extract_genotypes(VCF_GZ, "22", target_positions, N_SAMPLES)
    if len(matched_pos) == 0:
        print("  [retry] Trying contig name 'chr22'...")
        geno_matrix, matched_pos = extract_genotypes(VCF_GZ, "chr22", target_positions, N_SAMPLES)

    print(f"  [info] Matched {len(matched_pos)} SNPs × {geno_matrix.shape[1]} samples")

    if len(matched_pos) < 100:
        sys.exit(f"[error] Only {len(matched_pos)} SNPs matched — need at least 100.")

    # 3. Align weights
    pos_to_weight = dict(zip(pgs_use["pos"].astype(int), pgs_use["weight"]))
    weights = np.array([pos_to_weight.get(p, 0.0) for p in matched_pos])

    # 4. Benchmark
    print("\n  [benchmark] Running plaintext vs BFV-FHE...\n")
    records = []
    for n in SNP_COUNTS:
        n = min(n, len(matched_pos))
        print(f"    SNP={n:4d} ...", end=" ", flush=True)
        r = benchmark(geno_matrix, weights, n)
        records.append(r)
        print(f"Plaintext {r['plaintext_ms']}ms | FHE {r['fhe_ms']}ms | "
              f"Overhead {r['overhead_x']}× | PRS corr {r['prs_correlation']}")

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n  [saved] {RESULTS_CSV}")
    print("\n" + df.to_string(index=False))

    plot_results(df)
    print("\n[done] Experiment 1 complete.")

if __name__ == "__main__":
    main()
