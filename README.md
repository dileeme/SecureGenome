# Genomic Beacons Under Siege
### Quantifying Re-Identification Risk from Summary Statistics and the Performance Cost of Homomorphic Encryption

> **ACM CCS 2026 Submission #5286**
> **Dataset:** 1000 Genomes Project Phase 3, Chromosome 22 (N = 2,504)  
> **FHE Library:** TenSEAL (CKKS scheme)

---

## Project Structure

The repository is organized into three primary experimental modules, supported by processed genomic feature matrices and result tracking.

```text
├── experiments/
│   ├── benchmark.py              # [Exp 2] Baseline FHE PRS Benchmarking
│   ├── reidentification.py       # [Exp 1] MIA Phase Transition Analysis
│   └── tunedbenchmark.py         # [Exp 3] L=1 Parameter Compaction Optimization
│
├── figures/
│   ├── fhe_performance_overhead.png
│   ├── latency_optimization.png
│   ├── reid_auc_final_plot.png
│   ├── storage_optimization.png
│
├── results/
│   ├── benchmark_results.csv          # Standard FHE metrics (Exp 2)
│   ├── reidentification_results.csv   # Empirical AUC data (Exp 1)
│   └── tuned_benchmark_results.csv    # Optimized FHE metrics (Exp 3)
│
└── README.md
```
## Overview

This repository contains the full experimental pipeline for the paper, characterizing the "privacy cliff" in genomic beacons and providing a domain-optimized FHE mitigation via parameter compaction.

| # | Experiment | Key Result |
|---|-----------|------------|
| 1 | **Membership Inference Attack (MIA)** | AUC = 0.987 at $k=100$; identified sharp phase transition. |
| 2 | **Baseline FHE Benchmarking** | 21,569× overhead, ~111s latency, MAE = $1.52 \times 10^{-6}$. |
| 3 | **Parameter Compaction** | 56.4% latency reduction via domain-specific $L=1$ depth matching. |

---

## Setup & Requirements

### System Dependencies
- **OS:** Ubuntu 20.04+ or WSL2
- **Python:** 3.10+
- **RAM:** $\geq$ 8 GB (16 GB recommended for full-scale FHE benchmarks)

```bash
# Install system tools
sudo apt update && sudo apt install -y tabix bgzip python3-pip python3-venv

# Initialize Virtual Environment
python3 -m venv venv
source venv/bin/activate

# Install Python Libraries
pip install tenseal pandas numpy scikit-learn matplotlib tqdm
