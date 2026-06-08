"""Tests for the dissimilarity metric functions and prior transformations.

Metric functions live in :mod:`cassiopeia.dissimilarity`.
"""

import numpy as np
import pytest

from cassiopeia import dissimilarity
from cassiopeia.mixins import PriorTransformationError
from cassiopeia.utils import _transform_priors


@pytest.fixture
def data():
    s1 = np.array([0, 1, 0, -1, 1, 2])
    s2 = np.array([1, 1, 0, 0, 1, 3])
    priors = {
        0: {1: 0.5, 2: 0.5},
        1: {1: 0.5, 2: 0.5},
        2: {1: 0.25, 2: 0.75},
        3: {1: 0.3, 2: 0.7},
        4: {1: 0.4, 2: 0.6},
        5: {1: 0.1, 2: 0.05, 3: 0.85},
    }
    return {
        "s1": s1,
        "s2": s2,
        "all_missing": np.array([-1, -1, -1, -1, -1, -1]),
        "ambiguous": [(0,), (-1, 0), (0,), (-1, 0), (1,), (1,)],
        "ambiguous_no_missing": [(0,), (1, 0), (0,), (2, 0), (1,), (1,)],
        "priors": priors,
        "nlweights": _transform_priors(priors, "negative_log"),
        "iweights": _transform_priors(priors, "inverse"),
        "sqiweights": _transform_priors(priors, "square_root_inverse"),
    }


# ── Prior transformations ─────────────────────────────────────────────────────


def test_bad_prior_transformations():
    with pytest.raises(PriorTransformationError):
        _transform_priors({0: {1: 0}, 1: {1: -1, 2: -1.5}}, "negative_log")


def test_negative_log_prior_transformations(data):
    priors = data["priors"]
    expected = {c: {s: -np.log(p) for s, p in states.items()} for c, states in priors.items()}
    assert data["nlweights"] == expected


def test_inverse_prior_transformations(data):
    priors = data["priors"]
    expected = {c: {s: 1 / p for s, p in states.items()} for c, states in priors.items()}
    assert data["iweights"] == expected


def test_sq_inverse_prior_transformations(data):
    priors = data["priors"]
    expected = {c: {s: np.sqrt(1 / p) for s, p in states.items()} for c, states in priors.items()}
    assert data["sqiweights"] == expected


# ── weighted_hamming (ignores missing) ────────────────────────────────────────


def test_weighted_hamming_identical(data):
    assert dissimilarity.weighted_hamming(data["s1"], data["s1"]) == 0


def test_weighted_hamming_no_priors(data):
    assert dissimilarity.weighted_hamming(data["s1"], data["s2"]) == 3 / 5


def test_weighted_hamming_no_priors_unmodified_state():
    # Without weights the uncut state is configurable via unmodified_state.
    s1 = [0, 1, 5]
    s2 = [1, 2, 5]
    assert dissimilarity.weighted_hamming(s1, s2) == 3 / 3  # default uncut=0
    assert dissimilarity.weighted_hamming(s1, s2, unmodified_state=1) == 2 / 3


def test_weighted_hamming_priors_negative_log(data):
    result = dissimilarity.weighted_hamming(data["s1"], data["s2"], weights=data["nlweights"])
    priors = data["priors"]
    expected = np.sum([-np.log(priors[0][1]), -(np.log(priors[5][2]) + np.log(priors[5][3]))])
    assert result == expected / 5


def test_weighted_hamming_priors_inverse(data):
    result = dissimilarity.weighted_hamming(data["s1"], data["s2"], weights=data["iweights"])
    priors = data["priors"]
    expected = np.sum([1 / priors[0][1], 1 / priors[5][2] + 1 / priors[5][3]])
    assert result == expected / 5


def test_weighted_hamming_priors_sq_inverse(data):
    result = dissimilarity.weighted_hamming(data["s1"], data["s2"], weights=data["sqiweights"])
    priors = data["priors"]
    expected = np.sum(
        [np.sqrt(1 / priors[0][1]), np.sqrt(1 / priors[5][2]) + np.sqrt(1 / priors[5][3])]
    )
    assert result == expected / 5


def test_weighted_hamming_all_missing(data):
    assert (
        dissimilarity.weighted_hamming(data["s1"], data["all_missing"], weights=data["nlweights"])
        == 0
    )


# ── hamming (counts all disagreements, missing included) ──────────────────────


def test_hamming(data):
    # positions 0, 3, 5 disagree (the missing at position 3 counts)
    assert dissimilarity.hamming(data["s1"], data["s2"]) == 3


def test_hamming_identical(data):
    assert dissimilarity.hamming(data["s1"], data["s1"]) == 0


def test_hamming_all_missing(data):
    # s1 disagrees with an all-missing vector everywhere except position 3 (-1)
    assert dissimilarity.hamming(data["s1"], data["all_missing"]) == 5


# ── nonmissing_hamming (ignores missing, normalized over present) ─────────────


def test_nonmissing_hamming(data):
    # s1=[0,1,0,-1,1,2], s2=[1,1,0,0,1,3]; position 3 is skipped (missing).
    # pos0 (0 vs 1) involves the uncut state 0 -> +1; pos5 (2 vs 3) is a full
    # mismatch -> +2; the rest match. Normalized over 5 present positions.
    assert dissimilarity.nonmissing_hamming(data["s1"], data["s2"]) == 3 / 5


def test_nonmissing_hamming_scoring_and_unmodified_state():
    # mismatch -> +2, mismatch with the unmodified state -> +1, identical -> +0.
    s1 = [0, 1, 5]
    s2 = [1, 2, 5]
    # default unmodified_state=0: pos0 (0 vs 1) -> +1, pos1 (1 vs 2) -> +2,
    # pos2 (5 vs 5) -> +0; total 3 over 3 present.
    assert dissimilarity.nonmissing_hamming(s1, s2) == 3 / 3
    # with unmodified_state=1: pos0 (0 vs 1) -> +1 (1 is uncut), pos1 (1 vs 2)
    # -> +1 (1 is uncut), pos2 -> +0; total 2 over 3.
    assert dissimilarity.nonmissing_hamming(s1, s2, unmodified_state=1) == 2 / 3


def test_nonmissing_hamming_all_missing(data):
    assert dissimilarity.nonmissing_hamming(data["s1"], data["all_missing"]) == 0


# ── cluster dissimilarities (ambiguous states) ────────────────────────────────


def test_cluster_dissimilarity(data):
    result = dissimilarity.cluster_dissimilarity(
        dissimilarity.weighted_hamming,
        data["s1"],
        data["ambiguous"],
        -1,
        data["nlweights"],
        np.mean,
    )
    np.testing.assert_almost_equal(result, 1.2544, decimal=4)


def test_cluster_weighted_hamming(data):
    result = dissimilarity.cluster_weighted_hamming(
        data["s1"], data["ambiguous_no_missing"], -1, None
    )
    np.testing.assert_almost_equal(result, 0.4, decimal=4)

    result = dissimilarity.cluster_weighted_hamming(data["s1"], data["ambiguous"], -1, None)
    np.testing.assert_almost_equal(result, 0.4444, decimal=4)


# ── string resolution ─────────────────────────────────────────────────────────


def test_resolve_dissimilarity_by_name():
    fn = dissimilarity._resolve_dissimilarity("nonmissing_hamming")
    assert fn is dissimilarity.nonmissing_hamming


def test_resolve_dissimilarity_unknown_raises():
    with pytest.raises(ValueError):
        dissimilarity._resolve_dissimilarity("not_a_metric")


def test_removed_functions_absent(data):
    for name in (
        "hamming_distance",
        "nonmissing_hamming_distance",
        "weighted_hamming_distance",
        "hamming_similarity_without_missing",
        "weighted_hamming_similarity",
        "exponential_negative_hamming_distance",
        "cluster_dissimilarity_weighted_hamming_distance_min_linkage",
    ):
        assert not hasattr(dissimilarity, name)
