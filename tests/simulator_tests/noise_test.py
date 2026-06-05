"""Tests for the noise() character-matrix sequencing-error simulator."""

import numpy as np
import pytest

from cassiopeia.simulator import noise


def _as_str(matrix):
    return matrix.astype(object).to_numpy()


def test_zero_error_rate_is_noop(char_tdata):
    before = _as_str(char_tdata.obsm["characters"]).copy()
    with pytest.warns(UserWarning):
        noise(char_tdata, error_rate=0.0)
    after = _as_str(char_tdata.obsm["characters"])
    np.testing.assert_array_equal(before, after)


def test_full_error_changes_every_called_site(char_tdata):
    missing = "-"
    before = _as_str(char_tdata.obsm["characters"]).copy()
    noise(char_tdata, error_rate=1.0, random_seed=0)
    after = _as_str(char_tdata.obsm["characters"])

    for i in range(before.shape[0]):
        for j in range(before.shape[1]):
            if before[i, j] == missing:
                # Missing sites are untouched.
                assert after[i, j] == missing
            else:
                # Every called site is changed to a different, non-missing state.
                assert after[i, j] != before[i, j]
                assert after[i, j] != missing


def test_no_missing_introduced(char_tdata):
    noise(char_tdata, error_rate=1.0, random_seed=1)
    after = _as_str(char_tdata.obsm["characters"])
    # Only the two pre-existing missing entries remain.
    assert (after == "-").sum() == 2


def test_deterministic(char_tdata):
    other = char_tdata.copy()
    noise(char_tdata, error_rate=0.5, random_seed=42)
    noise(other, error_rate=0.5, random_seed=42)
    np.testing.assert_array_equal(
        _as_str(char_tdata.obsm["characters"]), _as_str(other.obsm["characters"])
    )


def test_key_added_preserves_original(char_tdata):
    before = _as_str(char_tdata.obsm["characters"]).copy()
    noise(char_tdata, error_rate=1.0, key_added="noisy", random_seed=0)
    # Original untouched, new key written.
    np.testing.assert_array_equal(_as_str(char_tdata.obsm["characters"]), before)
    assert "noisy" in char_tdata.obsm
    assert _as_str(char_tdata.obsm["noisy"]).shape == before.shape


def test_copy_leaves_input_untouched(char_tdata):
    before = _as_str(char_tdata.obsm["characters"]).copy()
    out = noise(char_tdata, error_rate=1.0, random_seed=0, copy=True)
    np.testing.assert_array_equal(_as_str(char_tdata.obsm["characters"]), before)
    assert not np.array_equal(_as_str(out.obsm["characters"]), before)


def test_imputed_priors_when_none_available(char_tdata):
    # Remove uns priors so noise must impute from the matrix.
    del char_tdata.uns["priors"]
    noise(char_tdata, error_rate=1.0, random_seed=0)
    after = _as_str(char_tdata.obsm["characters"])
    # Imputed states are drawn from observed non-missing states only.
    observed = set(np.unique(after)) - {"-"}
    assert observed.issubset({"0", "1", "2"})
