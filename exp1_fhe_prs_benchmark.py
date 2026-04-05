"""
Experiment 1: BFV-FHE vs Plaintext PRS Benchmark
Dataset:  1000 Genomes Phase 3 chr22 (local VCF)
Weights:  PGS000018 T2D score (local)
Output:   exp1_results.csv, exp1_plot.png
"""

import os, sys, gzip, io, math, time, tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import openfhe
from cyvcf2 import VCF

# ── HARDCODED PATHS (edit if your filenames differ) ───────────────────────────
VCF_PATH = "./data/chr22_1kg.vcf.gz"
PGS_PATH = "./data/PGS000018_weights.txt.gz"
# ─────────────────────────────────────────────────────────────────────────────

SNP_COUNTS = [100, 250, 500, 998]
N_SAMPLES  = 50


def load_pgs(path):
    with gzip.open(path, "rt") as f:
        lines = [l for l in f if not l.startswith("#")]
    df = pd.read_csv(io.StringIO("".join(lines)), sep="\t", low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"chr_name": "chrom", "chr_position": "pos",
                             "effect_allele": "ea", "effect_weight": "weight"})
    df["chrom"]  = df["chrom"].astype(str).str.replace("^chr", "", regex=True)
    df["pos"]    = pd.to_numeric(df["pos"],    errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["chrom", "pos", "weight"])
    chr22 = df[df["chrom"] == "22"].reset_index(drop=True)
    print(f"  PGS chr22 SNPs: {len(chr22)}")
    return chr22


def load_genotypes(vcf_path, positions_set, n_samples=50):
    vcf = VCF(vcf_path)
    n   = min(n_samples, len(vcf.samples))
    G, P = [], []
    for v in vcf("22"):
        if v.POS not in positions_set:
            continue
        gt = v.gt_types          # 0=HOM_REF 1=HET 2=HOM_ALT 3=UNKNOWN
        d  = np.where(gt == 3, np.nan, gt.astype(float))[:n]
        G.append(d); P.append(v.POS)
        if len(P) >= max(SNP_COUNTS):
            break
    vcf.close()
    print(f"  Matched {len(P)} SNPs × {n} samples")
    return np.array(G), P          # shape (n_snps, n_samples)


def prs_plain(G, w):
    mu = np.nanmean(G, axis=1, keepdims=True)
    G2 = np.where(np.isnan(G), mu, G)
    return w @ G2                  # (n_samples,)


def prs_bfv(G, w):
    n_snps, n_samp = G.shape
    mu = np.nanmean(G, axis=1, keepdims=True)
    G2 = np.where(np.isnan(G), mu, G)

    # batch MUST be power of 2
    batch = 2 ** int(math.floor(math.log2(min(n_snps, 4096))))

    SCALE = 100
    w_int = [int(round(x * SCALE)) for x in w[:batch]]
    w_int += [0] * (batch - len(w_int))

    # build context once per call
    p = openfhe.CCParamsBFVRNS()
    p.SetPlaintextModulus(786433)
    p.SetMultiplicativeDepth(1)
    p.SetBatchSize(batch)
    p.SetSecurityLevel(openfhe.SecurityLevel.HEStd_128_classic)

    cc = openfhe.GenCryptoContext(p)
    cc.Enable(openfhe.PKESchemeFeature.PKE)
    cc.Enable(openfhe.PKESchemeFeature.KEYSWITCH)
    cc.Enable(openfhe.PKESchemeFeature.LEVELEDSHE)

    kp = cc.KeyGen()
    cc.EvalSumKeyGen(kp.secretKey)
    cc.EvalMultKeyGen(kp.secretKey)

    pt_w = cc.MakePackedPlaintext(w_int)
    out  = []

    for s in range(n_samp):
        g_int = [int(round(G2[i, s] * SCALE)) for i in range(batch)]
        ct    = cc.Encrypt(kp.publicKey, cc.MakePackedPlaintext(g_int))
        ct    = cc.EvalMult(ct, pt_w)
        ct    = cc.EvalSum(ct, batch)
        dec   = cc.Decrypt(ct, kp.secretKey)
        dec.SetLength(1)
        out.append(dec.GetPackedValue()[0] / (SCALE * SCALE))

    return out


def run_benchmark(G, w, n):
    Gs = G[:n]; ws = w[:n]

    tracemalloc.start()
    t0 = time.perf_counter()
    p_plain = prs_plain(Gs, ws)
    t_plain = (time.perf_counter() - t0) * 1000
    _, mem_plain = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    t0 = time.perf_counter()
    p_fhe = prs_bfv(Gs, ws)
    t_fhe = (time.perf_counter() - t0) * 1000
    _, mem_fhe = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    corr = float(np.corrcoef(p_plain, p_fhe)[0, 1])
    return {
        "n_snps":           n,
        "plaintext_ms":     round(t_plain, 2),
        "fhe_ms":           round(t_fhe, 2),
        "overhead_x":       round(t_fhe / max(t_plain, 0.001), 1),
        "plaintext_mem_mb": round(mem_plain / 1e6, 3),
        "fhe_mem_mb":       round(mem_fhe   / 1e6, 3),
        "prs_correlation":  round(corr, 4),
    }


def plot(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("BFV-FHE vs Plaintext PRS Overhead\n"
                 "1000 Genomes chr22 · PGS000018 (T2D) · 128-bit security",
                 fontweight="bold", fontsize=10)

    fmt = ticker.FuncFormatter(lambda x, _: f"{int(x):,}")

    ax = axes[0]
    ax.plot(df.n_snps, df.plaintext_ms, "o-", color="#2ecc71", label="Plaintext")
    ax.plot(df.n_snps, df.fhe_ms,       "s-", color="#e74c3c", label="BFV-FHE")
    ax.set_yscale("log"); ax.set_xlabel("SNP count"); ax.set_ylabel("Time (ms)")
    ax.set_title("Latency"); ax.legend(); ax.xaxis.set_major_formatter(fmt)

    ax = axes[1]
    ax.bar(df.n_snps.astype(str), df.overhead_x, color="#3498db")
    for i, v in enumerate(df.overhead_x):
        ax.text(i, v + 1, f"{v}×", ha="center", fontsize=9)
    ax.set_xlabel("SNP count"); ax.set_ylabel("Overhead (×)")
    ax.set_title("Compute Overhead Multiplier")

    ax = axes[2]
    ax.plot(df.n_snps, df.plaintext_mem_mb, "o-", color="#2ecc71", label="Plaintext")
    ax.plot(df.n_snps, df.fhe_mem_mb,       "s-", color="#e74c3c", label="BFV-FHE")
    ax.set_xlabel("SNP count"); ax.set_ylabel("Peak memory (MB)")
    ax.set_title("Memory Footprint"); ax.legend(); ax.xaxis.set_major_formatter(fmt)

    plt.tight_layout()
    plt.savefig("exp1_plot.png", dpi=150, bbox_inches="tight")
    print("  Saved: exp1_plot.png")


def main():
    print("\n=== Experiment 1: FHE PRS Benchmark ===\n")

    for p in [VCF_PATH, PGS_PATH]:
        if not os.path.exists(p):
            sys.exit(f"[error] File not found: {p}")

    pgs  = load_pgs(PGS_PATH)
    pos_set = set(pgs["pos"].dropna().astype(int))

    print("\n  Loading genotypes from VCF...")
    G, matched = load_genotypes(VCF_PATH, pos_set, N_SAMPLES)

    if len(matched) < 100:
        sys.exit(f"[error] Only {len(matched)} SNPs matched — need ≥100")

    p2w = dict(zip(pgs["pos"].astype(int), pgs["weight"]))
    w   = np.array([p2w.get(p, 0.0) for p in matched])

    print("\n  Benchmarking...\n")
    rows = []
    for n in SNP_COUNTS:
        n = min(n, len(matched))
        print(f"  SNP={n} ...", end=" ", flush=True)
        r = run_benchmark(G, w, n)
        rows.append(r)
        print(f"plain={r['plaintext_ms']}ms  fhe={r['fhe_ms']}ms  "
              f"overhead={r['overhead_x']}×  corr={r['prs_correlation']}")

    df = pd.DataFrame(rows)
    df.to_csv("exp1_results.csv", index=False)
    print("\n" + df.to_string(index=False))
    plot(df)
    print("\n[done]")


if __name__ == "__main__":
    main()
