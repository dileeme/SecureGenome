import numpy as np
from typing import Sequence


def shringarpure_bustamante_lrt(
    maf_vector: np.ndarray,
    target_genotype: np.ndarray,
) -> float:
    p = np.clip(maf_vector, 1e-6, 1 - 1e-6)
    g = np.asarray(target_genotype, dtype=float)

    def _diploid_log_prob(dosage, freq):
        q = 1.0 - freq
        probs = np.where(
            dosage == 0, q**2,
            np.where(dosage == 1, 2 * freq * q, freq**2)
        )
        return np.log(np.clip(probs, 1e-300, None))

    N = 2504
    p_with = (2 * N * p + g) / (2 * N + 2)
    p_with = np.clip(p_with, 1e-6, 1 - 1e-6)

    ll_h1 = _diploid_log_prob(g, p_with)
    ll_h0 = _diploid_log_prob(g, p)

    return float(np.sum(ll_h1 - ll_h0))


def raisaro_score(
    maf_vector: np.ndarray,
    target_genotype: np.ndarray,
) -> float:
    p = np.clip(maf_vector, 1e-6, 1 - 1e-6)
    g = np.asarray(target_genotype, dtype=float)
    log_odds = np.log(p / (1.0 - p))
    return float(np.dot(g, log_odds))


def compute_baseline_aucs(
    X: np.ndarray,
    y: np.ndarray,
    maf_vector: np.ndarray,
) -> dict:
    from sklearn.metrics import roc_auc_score

    sb_scores = np.array([shringarpure_bustamante_lrt(maf_vector, row) for row in X])
    ra_scores = np.array([raisaro_score(maf_vector, row) for row in X])

    return {
        "sb_lrt_auc": roc_auc_score(y, sb_scores),
        "raisaro_auc": roc_auc_score(y, ra_scores),
    }
