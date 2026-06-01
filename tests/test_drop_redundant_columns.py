"""
file: test_drop_redundant_columns.py
---
Tests for the redundancy-reduction behavior introduced by the
``drop_redundant_columns`` option on the FeatureBuilder.

The reducer (``FeatureBuilder.generate_summary_stats``) only inspects the
*generated* numeric feature columns -- columns present in the original input
(``self.orig_data``) are always preserved untouched. The ``drop_redundant_columns``
flag itself only gates whether ``featurize`` *applies* the reduced frame:

    df_reduced = self.generate_summary_stats(self.chat_data)
    if self.drop_redundant_columns:
        self.chat_data = df_reduced

These tests use a fixture dataset (data/cleaned_data/test_redundant_columns.csv)
with deliberately redundant generated-feature columns:

    feat_base         monotonic 1..20
    feat_corr         = 2 * feat_base (Spearman corr 1.0 with feat_base; has 2 NaNs)
    feat_independent  uncorrelated zigzag (Spearman ~0.08 with feat_base)
    sparse_zeros      95% zeros  -> exceeds min_zero_ratio (0.9)
    mostly_na         50% NaN    -> exceeds min_na_ratio (0.3)
"""

import logging

import pandas as pd
import pytest

from team_comm_tools import FeatureBuilder

# The four columns that stand in for the original/input data. Everything else in
# the CSV is treated as a generated feature column eligible for reduction.
ORIGINAL_COLS = ["conversation_num", "speaker_nickname", "message", "timestamp"]

redundant_df = pd.read_csv("data/cleaned_data/test_redundant_columns.csv")


def make_reducer(drop_redundant_columns):
    """
    Build a FeatureBuilder with only the attributes generate_summary_stats needs,
    bypassing the (heavy) __init__/featurize pipeline so the reduction logic can
    be exercised in isolation.
    """
    fb = FeatureBuilder.__new__(FeatureBuilder)
    fb.orig_data = redundant_df[ORIGINAL_COLS].copy()
    fb.min_na_ratio = 0.3
    fb.min_zero_ratio = 0.9
    fb.corr_thresh = 0.9
    fb.min_group_size = 2
    fb.treat_zero_as_na = True
    fb.drop_redundant_columns = drop_redundant_columns
    fb.logger = logging.getLogger("test_drop_redundant")
    fb.summ_logger = logging.getLogger("test_drop_redundant_summary")
    return fb


def test_no_drop_default_preserves_all_columns():
    """
    (a) Default no-drop behavior: with drop_redundant_columns=False, featurize keeps
    the original (un-reduced) frame, so every column -- including the redundant ones --
    survives. We mirror the exact gating used in FeatureBuilder.featurize.
    """
    fb = make_reducer(drop_redundant_columns=False)
    input_df = redundant_df.copy()

    df_reduced = fb.generate_summary_stats(input_df)
    kept = df_reduced if fb.drop_redundant_columns else input_df

    # Nothing is dropped when the flag is off.
    assert list(kept.columns) == list(redundant_df.columns)
    for col in ["feat_base", "feat_corr", "feat_independent", "sparse_zeros", "mostly_na"]:
        assert col in kept.columns


def test_generate_summary_stats_does_not_mutate_input():
    """
    The reducer must be non-mutating: it returns a new frame and leaves the input
    untouched. This is what makes the no-drop path above safe.
    """
    fb = make_reducer(drop_redundant_columns=False)
    input_df = redundant_df.copy()
    cols_before = list(input_df.columns)

    fb.generate_summary_stats(input_df)

    assert list(input_df.columns) == cols_before


def test_drop_enabled_reduces_redundant_columns():
    """
    (b) Opt-in drop behavior: with drop_redundant_columns=True, redundant generated
    columns are removed -- correlated duplicates, high-zero columns, and high-NA
    columns -- while the representative and uncorrelated features are retained.
    """
    fb = make_reducer(drop_redundant_columns=True)

    reduced = fb.generate_summary_stats(redundant_df.copy())

    # Correlated duplicate, sparse, and high-NA columns are dropped.
    assert "feat_corr" not in reduced.columns       # correlated with feat_base
    assert "sparse_zeros" not in reduced.columns     # >90% zeros
    assert "mostly_na" not in reduced.columns        # >30% NA

    # The group representative and the uncorrelated feature are kept.
    assert "feat_base" in reduced.columns
    assert "feat_independent" in reduced.columns


def test_original_columns_always_preserved_when_dropping():
    """
    Even with dropping enabled, original/input columns are never analyzed or removed.
    """
    fb = make_reducer(drop_redundant_columns=True)

    reduced = fb.generate_summary_stats(redundant_df.copy())

    for col in ORIGINAL_COLS:
        assert col in reduced.columns


def test_correlated_group_keeps_best_representative():
    """
    Within a correlated group, the representative with the most valid data (and
    highest variance as a tiebreak) is kept. feat_base has 20 valid values; feat_corr
    has 18 (two NaNs), so feat_base must be the survivor and feat_corr dropped.
    """
    fb = make_reducer(drop_redundant_columns=True)

    reduced = fb.generate_summary_stats(redundant_df.copy())

    assert "feat_base" in reduced.columns
    assert "feat_corr" not in reduced.columns


@pytest.mark.parametrize("sparse_col", ["sparse_zeros", "mostly_na"])
def test_sparse_columns_removed_only_when_enabled(sparse_col):
    """
    Sparse columns (high-zero / high-NA) are removed when dropping is enabled, but
    preserved (via the no-drop gating) when it is disabled.
    """
    # enabled -> removed
    fb_on = make_reducer(drop_redundant_columns=True)
    reduced = fb_on.generate_summary_stats(redundant_df.copy())
    assert sparse_col not in reduced.columns

    # disabled -> preserved (featurize keeps the original frame)
    fb_off = make_reducer(drop_redundant_columns=False)
    input_df = redundant_df.copy()
    df_reduced = fb_off.generate_summary_stats(input_df)
    kept = df_reduced if fb_off.drop_redundant_columns else input_df
    assert sparse_col in kept.columns
