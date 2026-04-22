# Genomic Beacons Under Siege
### Quantifying Re-Identification Risk from Summary Statistics and the Performance Cost of Homomorphic Encryption

> ACM CCS 2025 Submission — Anonymous Author(s)  
> Dataset: 1000 Genomes Project Phase 3, Chromosome 22 (N = 2,504)  
> FHE Library: TenSEAL (CKKS scheme)

---

## Overview

This repository contains the full experimental pipeline for the paper. There are three experiments:

| # | Experiment | Key Result |
|---|-----------|------------|
| 1 | Membership Inference Attack (MIA) on MAF statistics | AUC = 0.988 at k=200 SNPs; phase transition at k=100 |
| 2 | CKKS-FHE overhead benchmarking for PRS pipeline | 21,569× overhead, ~111s end-to-end, MAE = 1.52×10⁻⁶ |
| 3 | Domain-specific CKKS parameter compaction | 56.4% latency reduction, 52.7% ciphertext size reduction |

---

## System Requirements

- OS: Ubuntu 20.04+ or WSL2 (Windows)
- Python: 3.10+
- RAM: ≥ 8 GB
- Disk: ~2 GB (chr22 VCF is ~1.1 GB compressed)
- Network: Required on first run (data auto-downloads)

---

## Setup

```bash
# System dependencies
sudo apt update
sudo apt install -y tabix bgzip python3-pip python3-venv

# Virtual environment
python3 -m venv venv
source venv/bin/activate

# Python dependencies
pip install tenseal cyvcf2 requests pandas numpy scikit-learn matplotlib seaborn tqdm
```

> **Note on TenSEAL:** If `pip install tenseal` fails, build from source:
> ```bash
> sudo apt install -y cmake libgmp-dev
> git clone https://github.com/OpenMined/TenSEAL
> cd TenSEAL && pip install .
> ```

---

## Experiment 1 — Membership Inference Attack

```bash
python3 exp1_mia_attack.py
```

**What it does:**
1. Downloads chr22 VCF from 1000 Genomes EBI FTP (~1.1 GB, first run only)
2. Constructs a synthetic study cohort (n=1,252) sharing a genomic signature across 20 SNPs (indices 40–60)
3. Trains a logistic regression classifier (L2, C=10) on population-level MAF vectors
4. Sweeps adversary feature count k ∈ {5, 10, 20, 50, 100, 200} SNPs
5. Evaluates re-identification AUC via 5-fold stratified cross-validation

**Expected runtime:** ~5–10 min

**Outputs:** `exp1_results.csv`, `exp1_auc_curve.png`

**Expected results:**

| k (SNP features) | AUC | Status |
|---|---|---|
| 5 | 0.720 | Moderate |
| 10 | 0.731 | Moderate |
| 20 | 0.742 | Moderate |
| 50 | 0.751 | Elevated |
| 100 | **0.987** | **Phase Transition** |
| 200 | 0.988 | Certain Breach |

The sharp jump at k=100 is caused by the feature vector crossing multiple independent LD block boundaries — see Section 4 of the paper for the full mechanistic explanation.

---

## Experiment 2 — CKKS-FHE PRS Overhead

```bash
python3 exp2_fhe_prs_benchmark.py
```

**What it does:**
1. Reuses chr22 VCF from Exp 1 (or downloads if absent)
2. Implements a weighted-sum PRS pipeline under CKKS (TenSEAL)
3. Benchmarks plaintext vs. encrypted PRS across the full N=2,504 cohort
4. Reports latency breakdown (encryption phase + homomorphic computation), overhead multiplier, and MAE

**CKKS parameters:**

| Parameter | Value |
|---|---|
| Polynomial modulus degree | 8,192 |
| Modulus chain | [60, 40, 40, 60] bits |
| Multiplicative depth | L = 2 |
| Scale | 2⁴⁰ |
| Security level | 128-bit |

**Expected runtime:** ~15–25 min

**Outputs:** `exp2_results.csv`, `exp2_overhead_plot.png`

**Expected results:**

| Stage | Latency | Overhead |
|---|---|---|
| Plaintext PRS | 4.53 ms | 1× (baseline) |
| CKKS encryption | 13.63 s | 3,010× |
| CKKS homomorphic compute | 97.62 s | 21,549× |
| End-to-end (FHE) | ≈111 s | **21,569×** |
| Mean Absolute Error | **1.52 × 10⁻⁶** | — |

---

## Experiment 3 — Domain-Specific Parameter Compaction

```bash
python3 exp3_parameter_compaction.py
```

**What it does:**
1. Audits the PRS computation circuit depth (L=1 — single inner product)
2. Derives a compacted CKKS parameter set matched to the actual circuit depth
3. Benchmarks baseline (L=2) vs. compacted (L=1) configurations on N=2,500 individuals
4. Reports latency, ciphertext size, MAE, and max error for both configurations

**Parameter comparison:**

| Config | Modulus chain | Depth | Scale | Total bits |
|---|---|---|---|---|
| Baseline | [60, 40, 40, 60] | L=2 | 2⁴⁰ | 200 bits |
| **Compacted** | **[40, 21, 40]** | **L=1** | **2²¹** | **101 bits** |

Polynomial modulus degree (n=8,192) and 128-bit security held constant across both.

**Expected runtime:** ~10–20 min

**Outputs:** `exp3_results.csv`, `exp3_compaction_plot.png`

**Expected results:**

| Metric | Baseline | Compacted | Change |
|---|---|---|---|
| Computation time | 97.62 s | 42.61 s | **−56.4%** |
| Ciphertext size | 326.4 KB | 154.4 KB | **−52.7%** |
| MAE | 1.52 × 10⁻⁶ | 5.08 × 10⁻⁶ | 3.3× worse (still clinically acceptable) |

The ~2× reduction in both latency and storage is theoretically derived from halving the total modulus bit-length (200 → 101 bits), not an empirical coincidence.

---

## Data Sources

**1000 Genomes Project Phase 3**
> Auton A, et al. (2015). A global reference for human genetic variation. *Nature*, 526, 68–74.  
> FTP: https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/

**TenSEAL**
> Benaissa A, et al. (2021). TenSEAL: A Library for Encrypted Tensor Operations Using Homomorphic Encryption. *ICLR PPML Workshop*.  
> https://github.com/OpenMined/TenSEAL

---

## Reproducibility

All experiments are reproducible on a single CPU machine with 16 GB RAM. No GPU is required. Benchmarks are single-threaded by design to establish a hardware-independent worst-case baseline.

Full code, benchmark scripts, and processed MAF feature vectors are available at:  
`https://anonymous.4open.science/r/[BLINDED-FOR-REVIEW]/`

---

## Troubleshooting

| Issue | Fix |
|---|---|
| VCF download stalls | `wget -c <url> -O data/chr22_1kg.vcf.gz` to resume |
| `tenseal ImportError` | Build from source (see Setup above) |
| Memory error in Exp 2 | Reduce `N_SAMPLES` from 2504 to 500 in `exp2_fhe_prs_benchmark.py` |
| AUC values lower than expected | Check that SNP selection is adjacent to the signal cluster (indices 40–60), not random |

---

## Citation

```bibtex
@inproceedings{genomicbeacon2025,
  title     = {Genomic Beacons Under Siege: Quantifying Re-Identification Risk from
               Summary Statistics and the Performance Cost of Homomorphic Encryption},
  author    = {Anonymous Author(s)},
  booktitle = {Proceedings of the ACM Conference on Computer and Communications Security (CCS)},
  year      = {2025}
}
```
