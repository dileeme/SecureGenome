# Population-Matched Privacy Risk Quantification in Genomic Beacons and the Cost of Homomorphic Mitigation

> **IEEE BIBM 2026 Submission**
> **Dataset:** 1000 Genomes Project Phase 3, Chromosome 22 (N = 2,504)
> **FHE Library:** TenSEAL (CKKS scheme)

---

## Overview

| # | Experiment | Script(s) | Key Result |
|---|---|---|---|
| 1 | IBD relative-assisted MIA (main) | `ibd_attack.py`, `significance.py`, `stratification_check.py` | Significant at k=80–100, AUC ~0.55–0.56 (population-matched controls, n=300) |
| 1S | Ablation — external-label null | `attack.py` | AUC ≈ 0.50 at all k — confirms no circular leakage (Table S1) |
| 2 | FHE Overhead Baseline | `benchmark.py` | End-to-end overhead 29,268×; MAE 2.17×10⁻⁶ |
| 3 | Parameter Compaction + Factorial | `compaction.py`, `factorial.py` | Config B (L=1, scale=2⁴⁰) recommended; depth reduction drives −48.9% latency |

---

## Repository Structure

```
GenomicBeaconsUnderSiege/
├── requirements.txt
├── data/
│   ├── raw/          # 1000G chr22 VCF, panel file, pedigree file
│   └── processed/    # derived genotype matrix (TSV)
├── src/
│   ├── exp1_membership_inference/
│   │   ├── ibd_attack.py          # main IBD relative-assisted MIA
│   │   ├── significance.py        # permutation test + bootstrap CI (unmatched controls)
│   │   ├── stratification_check.py# population stratification check + matched-control rerun
│   │   ├── attack.py              # ablation: external-label null (AUC≈0.50, Table S1)
│   │   ├── baselines.py           # Shringarpure-Bustamante LRT, Raisaro score
│   │   └── cohort_construction.py # external superpopulation label construction
│   ├── exp2_fhe_overhead/
│   │   └── benchmark.py           # CKKS baseline overhead benchmark
│   ├── exp3_parameter_compaction/
│   │   ├── compaction.py          # individual config benchmarks
│   │   └── factorial.py           # 2×2 factorial: {L=1,L=2} × {scale=2²¹,2⁴⁰}
│   └── analysis/
│       └── cue.py                 # CUE table with cross-experiment sanity checks
├── results/
│   ├── exp1/
│   │   ├── reidentification_results_ibd.csv
│   │   ├── significance_results.csv           # unmatched controls
│   │   └── significance_results_matched_controls.csv
│   ├── exp2/
│   ├── exp3/
│   └── analysis/
├── figures/
└── tests/
    └── test_no_circularity.py     # regression guard against circular cohort construction
```

---

## Experiment Descriptions

### Experiment 1 — IBD Relative-Assisted Membership Inference (Main Result)

Attack model: 600 mothers from 1000G complete trios constitute the beacon study
cohort (label=1). Each child's genome is simulated via Mendelian inheritance from
the phased parental haplotypes — one allele from the father's phased haplotype,
one from the mother's — producing a simulated child genome sharing exactly 50% IBD
with the mother. The adversary holds the simulated child genome (representing a
DTC database leak of the biological child's genotype) and queries the beacon to
infer maternal membership. Controls (label=0) are unrelated individuals whose own
genotypes serve as the adversary reference, providing zero IBD signal.

**Three methods compared:** logistic regression (C=10, 5-fold stratified CV),
Shringarpure–Bustamante LRT, and Raisaro score function.

**SNP sweep:** k = {5, 10, 20, 50, 60, 70, 80, 90, 100, 110, 200}.

**Significance testing:** 1,000-permutation label-shuffle test + 1,000-sample
bootstrap 95% CI, implemented in `significance.py`.

**Population stratification check:** `stratification_check.py` verifies that
controls are population-matched to the trio mothers and re-runs the significance
pipeline with matched controls. Key findings:
- The random control draw was significantly mismatched (chi-square p ≈ 0.000);
  AMR is 24.8% of mothers but 4.0% of controls; EAS is 11.5% vs. 28.7%.
- Exact population matching is structurally limited to n=300 controls (the 1000G
  trio recruitment depletes the non-trio pool in populations like GWD, IBS, CHS).
- With matched controls: signal is significant at k=80–100 (AUC ~0.55–0.56)
  and is **not** significant at k=200 (AUC=0.515, p=0.276). The high-k tail in
  unmatched results reflects population structure, not IBD.
- The unmatched-control results (onset k=70, peak AUC=0.574 at k=200) are
  reported as the uncorrected baseline for comparison and the matched-control results
  are the primary claim.

**Ablation (Table S1, `attack.py`):** Labels derived from 1000G superpopulation
metadata (external, no SNP signal) produce AUC ≈ 0.50 at all k, confirming
absence of circular leakage.

---

### Experiment 2 — FHE Overhead Baseline

Benchmarks CKKS homomorphic encryption overhead for genomic PRS computation
at N=2,504 individuals × 200 SNPs, baseline parameter set (depth L=2,
scale 2⁴⁰, coeff_mod=[60,40,40,60]).

| Metric | Value | Overhead |
|---|---|---|
| Plaintext PRS computation | 3.0 ms | 1× (baseline) |
| CKKS encryption phase | 11.3 s | 3,767× |
| CKKS homomorphic computation | 76.5 s | 25,498× |
| **End-to-end (enc + comp)** | **87.8 s** | **29,268×** |
| Mean Absolute Error (MAE) | 2.17×10⁻⁶ | — |

Both overhead definitions are reported explicitly. The end-to-end figure (29,268×)
is the operationally relevant one and the figure cited in the abstract.

MAE of 2.17×10⁻⁶ confirms CKKS approximation error is negligible for PRS at
baseline parameters.

**Data note:** Benchmarks use synthetic genotype-shaped data
(`np.random.randint(0, 3, ...)`, shape N×K). CKKS timing is determined by
ciphertext structure and cryptographic parameters, not by plaintext values —
standard practice in FHE benchmarking. Set `USE_REAL_DATA=True` in `benchmark.py`
to use the real genotype matrix.

---

### Experiment 3 — Parameter Compaction and Factorial Decomposition

Full 2×2 factorial across {L=1, L=2} × {scale=2²¹, 2⁴⁰}, N=2,504 × 200 SNPs.

| Config | Depth (L) | Modulus chain | Scale | Comp. Time | CT Size | MAE |
|---|---|---|---|---|---|---|
| A — Baseline | 2 | [60,40,40,60] | 2⁴⁰ | 75.6 s | 326.6 KB | 5.4×10⁻⁶ |
| **B — Depth only** | **1** | **[60,40,60]** | **2⁴⁰** | **38.6 s** | **229.8 KB** | **4.6×10⁻⁷** |
| C — Scale only | 2 | [60,21,21,60] | 2²¹ | 75.0 s | 238.0 KB | 0.336 |
| D — Compacted | 1 | [40,21,40] | 2²¹ | 39.3 s | 155.6 KB | 0.183 |

**Config B is the recommended configuration.** Depth reduction (L=2→1) drives
−48.9% computation time while *improving* precision. Scale reduction (2⁴⁰→2²¹)
contributes −0.8% speed but causes all accuracy loss. Config D's MAE of 0.183
could alter PRS decile classifications and is clinically unsafe.

**Marginal attribution:**

| Effect | Δ Time | Δ Time (%) | Δ MAE |
|---|---|---|---|
| Depth reduction (A→B) | −36.9 s | −48.9% | −4.9×10⁻⁶ (improves) |
| Scale reduction (A→C) | −0.6 s | −0.8% | +0.336 |
| Interaction | +1.3 s | +1.7% | −0.153 |
| Total (A→D) | −36.2 s | −47.9% | +0.183 |

---

## Setup

### Requirements

- Python 3.10+
- RAM ≥ 8 GB (16 GB recommended for full-scale FHE benchmarks)

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Pinned versions:
```
tenseal==0.3.14, pandas==2.2.2, numpy==1.26.4,
scikit-learn==1.5.0, matplotlib==3.9.0, tqdm==4.66.4, scipy>=1.11
```

### Data Setup (Experiment 1 only)

Download the 1000 Genomes chr22 genotype matrix, panel file, and pedigree file:

```bash
# Panel file (sample -> superpopulation mapping)
wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel \
    -O data/raw/integrated_call_samples_v3.20130502.ALL.panel

# Pedigree file (trio family structure)
wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/working/20130606_sample_info/20130606_g1k.ped \
    -O data/raw/20130606_g1k.ped

# Genotype matrix (VCF → TSV conversion, chr22)
# See data/raw/README for conversion steps
```

---

## Reproducing Results

Run experiments in order (Exp 2 and 3 are independent; Exp 1 requires panel and pedigree files):

```bash
# Experiment 2 — FHE overhead baseline (no data needed)
python -m src.exp2_fhe_overhead.benchmark

# Experiment 3 — Full factorial decomposition (no data needed)
python -m src.exp3_parameter_compaction.factorial

# Experiment 1 — IBD attack (requires panel file, pedigree file, and genotype TSV)
python -m src.exp1_membership_inference.ibd_attack \
    --genotype-tsv data/processed/genotype_matrix.tsv \
    --ped data/raw/20130606_g1k.ped

# Experiment 1 — Significance test (unmatched controls)
python -m src.exp1_membership_inference.significance \
    --genotype-tsv data/processed/genotype_matrix.tsv \
    --ped data/raw/20130606_g1k.ped

# Experiment 1 — Population stratification check + matched-control rerun
python -m src.exp1_membership_inference.stratification_check \
    --genotype-tsv data/processed/genotype_matrix.tsv \
    --ped data/raw/20130606_g1k.ped \
    --panel data/raw/integrated_call_samples_v3.20130502.ALL.panel

# Experiment 1 — Ablation: external-label null (Table S1)
python -m src.exp1_membership_inference.attack \
    --genotype-tsv data/processed/genotype_matrix.tsv \
    --panel data/raw/integrated_call_samples_v3.20130502.ALL.panel \
    --superpopulation EUR

# CUE table (requires results from all three experiments)
python -m src.analysis.cue
```

### Running Tests

```bash
python -m pytest tests/ -v
```

The circularity regression test (`tests/test_no_circularity.py`) will fail loudly
if the deprecated circular cohort construction method is used in place of the
external label construction.
