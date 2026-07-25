import numpy as np
import pandas as pd


def construct_label_external(panel_path: str, superpopulation: str = "EUR", seed: int = 42):
    panel = pd.read_csv(panel_path, sep="\t")

    panel.columns = [c.lower() for c in panel.columns]

    sp_col = next(c for c in panel.columns if "super" in c)
    sample_col = next(c for c in panel.columns if "sample" in c)

    subset = panel[panel[sp_col] == superpopulation].copy()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(subset))
    half = len(subset) // 2
    labels = np.zeros(len(subset), dtype=int)
    labels[idx[:half]] = 1

    return subset[sample_col].tolist(), labels



def construct_label_circular_DEPRECATED(X_all: np.ndarray):
  
    import warnings
    warnings.warn(
        "construct_label_circular_DEPRECATED is for ablation only and must NOT "
        "be used in main experiments. It produces artificially inflated AUC via LD.",
        DeprecationWarning,
        stacklevel=2,
    )
    signal_snps = X_all[:, 40:60].sum(axis=1)
    threshold = np.percentile(signal_snps, 50)
    y = (signal_snps > threshold).astype(int)
    return y


CIRCULAR_LABEL_SNP_RANGE = (40, 60)
