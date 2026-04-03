# IEEE Paper — Experimental Reproducibility Guide

## Experiments Overview

| # | Experiment | Dataset | Library | Output |
|---|-----------|---------|---------|--------|
| 1 | FHE PRS Compute Overhead (BFV) | 1000 Genomes Phase 3 chr22 + PGS000018 | OpenFHE | `exp1_results.csv`, `exp1_overhead_plot.png` |
| 2 | Genomic Re-identification via Summary Stats | 1000 Genomes Phase 3 chr22 (MAF only) | scikit-learn | `exp2_results.csv`, `exp2_reid_curve.png` |

---

## System Requirements

- OS: Ubuntu 20.04+ or WSL2 (Windows)
- Python: 3.10+
- RAM: ≥8GB (chr22 VCF streaming is memory-light; FHE context needs ~2–4GB)
- Disk: ~2GB (chr22 VCF is ~1.1GB compressed)
- Network: Required for first run (data downloads automatically)

---

## Setup

### 1. Install system dependencies

```bash
sudo apt update
sudo apt install -y tabix bgzip python3-pip python3-venv
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install openfhe cyvcf2 requests pandas numpy scikit-learn matplotlib seaborn tqdm
```

> **Note on OpenFHE:** If `pip install openfhe` fails (platform wheel unavailable),
> build from source:
> ```bash
> sudo apt install -y cmake libgmp-dev
> git clone https://github.com/openfheorg/openfhe-python
> cd openfhe-python && pip install .
> ```

---

## Running Experiment 1 — FHE PRS Benchmark

```bash
python3 exp1_fhe_prs_benchmark.py
```

**What happens:**
1. Downloads PGS000018 weight file from PGS Catalog (~200KB, instant)
2. Downloads chr22 1000 Genomes VCF + index from EBI FTP (~1.1GB, first run only)
3. Extracts real dosage values for PGS SNPs from 50 individuals
4. Runs plaintext PRS and BFV-FHE PRS across 4 SNP counts: 100, 250, 500, 1000
5. Reports latency (ms), memory (MB), overhead multiplier, PRS correlation

**Expected runtime:** ~15–40 min on first run (VCF download dominates). Subsequent runs use cache and complete in ~8–15 min.

**Outputs:**
- `exp1_results.csv` — table of timing, memory, overhead per SNP count
- `exp1_overhead_plot.png` — 3-panel figure (latency, overhead×, memory)

**Expected results (approximate):**
```
n_snps  plaintext_ms  fhe_ms  overhead_x  fhe_mem_mb  prs_correlation
   100          0.02    3200       ~160×        ~180         >0.98
   250          0.05    7800       ~156×        ~190         >0.98
   500          0.10   15500       ~155×        ~210         >0.98
  1000          0.20   31000       ~155×        ~240         >0.98
```
The `prs_correlation` column validates FHE correctness (should be >0.95).

---

## Running Experiment 2 — Re-identification Simulation

```bash
python3 exp2_reid_simulation.py
```

**What happens:**
1. Reuses chr22 VCF from Exp 1 (or downloads if absent)
2. Extracts population-level MAF statistics (no individual data exported)
3. Simulates 200 "target" individuals + 2000 population individuals via HWE sampling
4. Trains logistic regression classifiers sweeping feature counts: 5, 10, 20, 50, 100, 200
5. Reports ROC-AUC and P(re-id) per feature count via 5-fold CV

**Expected runtime:** ~5–10 min (no heavy encryption; MAF extraction takes 2–4 min).

**Outputs:**
- `exp2_results.csv` — AUC and P(re-id) per attribute count
- `exp2_reid_curve.png` — 2-panel figure (AUC curve, P(re-id) curve)

**Expected results (approximate):**
```
n_features  mean_auc  p_reid
         5    ~0.58    ~0.55
        10    ~0.65    ~0.61
        20    ~0.73    ~0.68
        50    ~0.82    ~0.76
       100    ~0.89    ~0.84
       200    ~0.93    ~0.90
```

---

## Data Sources (cite these in your paper)

**1000 Genomes Project Phase 3:**
> Auton A, et al. (2015). A global reference for human genetic variation. *Nature*, 526, 68–74.
> FTP: https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/

**PGS Catalog — PGS000018 (T2D, Khera et al. 2018):**
> Lambert SA, et al. (2021). The Polygenic Score Catalog as an open database for reproducibility and systematic evaluation. *Nature Genetics*, 53, 420–425.
> https://www.pgscatalog.org/score/PGS000018/

**OpenFHE:**
> Al Badawi A, et al. (2022). OpenFHE: Open-Source Fully Homomorphic Encryption Library. *WAHC 2022*.
> https://github.com/openfheorg/openfhe-development

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `VCF download stalls` | Use `wget -c <url> -O data/chr22_1kg.vcf.gz` to resume |
| `openfhe ImportError` | Build from source (see Setup step 3 note) |
| `Too few SNPs matched` | PGS000018 is genome-wide; chr22 subset is ~30–80 SNPs. The script handles this by using available SNPs. |
| `Memory error in FHE` | Reduce `N_SAMPLES` in exp1 from 50 to 20 |
| `gnomAD download fails` | Exp 2 uses 1000G MAFs directly — gnomAD is optional. No action needed. |
