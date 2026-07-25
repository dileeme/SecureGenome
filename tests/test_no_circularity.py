import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.exp1_membership_inference.cohort_construction import (
    CIRCULAR_LABEL_SNP_RANGE,
    construct_label_circular_DEPRECATED,
)

LD_BUFFER = 20
LABEL_START, LABEL_END = CIRCULAR_LABEL_SNP_RANGE 


def feature_window_from_attack(k: int, feature_start: int = 0) -> range:
    return range(feature_start, feature_start + k)


def assert_no_overlap_or_adjacency(label_range: tuple, feature_range: range, ld_buffer: int = LD_BUFFER):
    l_start, l_end = label_range
    f_start = feature_range.start
    f_end = feature_range.stop 


    overlap = not (f_end <= l_start or f_start >= l_end)
    assert not overlap, (
        f"CIRCULARITY: feature range [{f_start},{f_end}) overlaps label range "
        f"[{l_start},{l_end}). This reproduces the Fix-1 bug."
    )

    too_close_left = f_end > l_start - ld_buffer and f_end <= l_start  
    too_close_right = f_start >= l_end and f_start < l_end + ld_buffer 
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
        safe_ks = [5, 10, 20]
        for k in safe_ks:
            fw = feature_window_from_attack(k, feature_start=0)
            assert_no_overlap_or_adjacency(CIRCULAR_LABEL_SNP_RANGE, fw)

    def test_original_circular_bug_would_fail(self):
        k = 5
        original_feature_range = range(60, 60 + k)
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
        import inspect
        from src.exp1_membership_inference.cohort_construction import construct_label_external
        sig = inspect.signature(construct_label_external)
        params = list(sig.parameters.keys())
        forbidden = {"X_all", "X", "genotype", "matrix", "snp_data", "data"}
        assert not forbidden.intersection(params), (
            f"construct_label_external has suspicious parameters: {params}. "
            "Label construction must be independent of genotype data."
        )
