import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.exp1_membership_inference.cohort_construction import (
    CIRCULAR_LABEL_SNP_RANGE,
    construct_label_circular_DEPRECATED,
)


# LD-adjacency buffer: SNPs within this many positions of the label window
# boundary are considered potentially LD-linked.
LD_BUFFER = 20

LABEL_START, LABEL_END = CIRCULAR_LABEL_SNP_RANGE  # (40, 60) — for reference only


def feature_window_from_attack(k: int, feature_start: int = 0) -> range:
    """Returns the SNP index range used as features in attack.py."""
    return range(feature_start, feature_start + k)


def assert_no_overlap_or_adjacency(label_range: tuple, feature_range: range, ld_buffer: int = LD_BUFFER):
    """
    Asserts that feature_range does not overlap or come within ld_buffer
    positions of the label range boundaries.
    """
    l_start, l_end = label_range
    f_start = feature_range.start
    f_end = feature_range.stop  # exclusive

    # Direct overlap
    overlap = not (f_end <= l_start or f_start >= l_end)
    assert not overlap, (
        f"CIRCULARITY: feature range [{f_start},{f_end}) overlaps label range "
        f"[{l_start},{l_end}). This reproduces the Fix-1 bug."
    )

    # LD-adjacency: the gap between feature window and label window is < ld_buffer.
    # "Within ld_buffer positions" means the gap must be strictly less than ld_buffer.
    too_close_left = f_end > l_start - ld_buffer and f_end <= l_start  # gap < ld_buffer
    too_close_right = f_start >= l_end and f_start < l_end + ld_buffer  # gap < ld_buffer
    assert not too_close_left, (
        f"LD-ADJACENCY: feature range [{f_start},{f_end}) is within {ld_buffer} SNPs "
        f"of label range left boundary {l_start}. Risk of LD-driven AUC inflation."
    )
    assert not too_close_right, (
        f"LD-ADJACENCY: feature range [{f_start},{f_end}) starts within {ld_buffer} SNPs "
        f"of label range right boundary {l_end}. Risk of LD-driven AUC inflation."
    )


class TestNoCircularity:

    def test_attack_feature_windows_do_not_overlap_circular_label_range(self):
        """
        The feature windows used in attack.py (columns 0:k) must not overlap
        or be LD-adjacent to the deprecated circular label range (cols 40-60).

        attack.py uses feature_start=0, so feature window is [0, k).
        The circular label range is [40, 60). The feature window [0, k) is
        safe as long as k <= 40 - LD_BUFFER = 20.

        Since attack.py now uses external (metadata-derived) labels, the
        positional constraint is no longer functionally required — but this
        test documents that the *deprecated* circular label method would be
        unsafe for k > 20 with the current feature start of 0.
        """
        safe_ks = [5, 10, 20]
        for k in safe_ks:
            fw = feature_window_from_attack(k, feature_start=0)
            # For safe k values, no adjacency to label range
            assert_no_overlap_or_adjacency(CIRCULAR_LABEL_SNP_RANGE, fw)

    def test_original_circular_bug_would_fail(self):
        """
        Reproduce the original circular construction to confirm it WOULD
        fail our overlap/adjacency check. Feature window was [60, 60+k),
        which immediately follows the label window [40, 60) — LD-adjacent.
        """
        k = 5
        original_feature_range = range(60, 60 + k)  # original code: X_all[:, 60:60+k]
        with pytest.raises(AssertionError, match="LD-ADJACENCY"):
            assert_no_overlap_or_adjacency(CIRCULAR_LABEL_SNP_RANGE, original_feature_range)

    def test_deprecated_label_function_emits_warning(self):
        """construct_label_circular_DEPRECATED must emit a DeprecationWarning."""
        import warnings
        X_dummy = np.random.randint(0, 3, size=(100, 100))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            construct_label_circular_DEPRECATED(X_dummy)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "ablation" in str(w[0].message).lower()

    def test_external_label_has_no_snp_range_dependence(self):
        """
        The external label function signature does not accept a genotype matrix
        or any SNP index — it takes only a panel_path, superpopulation, and seed.
        This structurally prevents any positional SNP dependence.
        """
        import inspect
        from src.exp1_membership_inference.cohort_construction import construct_label_external
        sig = inspect.signature(construct_label_external)
        params = list(sig.parameters.keys())
        # Must not have any parameter that could accept a numpy array / genotype matrix
        forbidden = {"X_all", "X", "genotype", "matrix", "snp_data", "data"}
        assert not forbidden.intersection(params), (
            f"construct_label_external has suspicious parameters: {params}. "
            "Label construction must be independent of genotype data."
        )
