"""Tests for the dissimilarity metric functions and prior transformations.

Metric functions now live in :mod:`cassiopeia.dissimilarity`.
"""

import numpy as np
import pytest

from cassiopeia import dissimilarity
from cassiopeia.solver import solver_utilities


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
        "nlweights": solver_utilities.transform_priors(priors, "negative_log"),
        "iweights": solver_utilities.transform_priors(priors, "inverse"),
        "sqiweights": solver_utilities.transform_priors(priors, "square_root_inverse"),
    }


# ── Prior transformations ─────────────────────────────────────────────────────


def test_bad_prior_transformations():
    with pytest.raises(solver_utilities.PriorTransformationError):
        solver_utilities.transform_priors({0: {1: 0}, 1: {1: -1, 2: -1.5}}, "negative_log")


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


# ── weighted_hamming_distance ─────────────────────────────────────────────────


def test_weighted_hamming_distance_identical(data):
    assert dissimilarity.weighted_hamming_distance(data["s1"], data["s1"]) == 0


def test_weighted_hamming_distance_no_priors(data):
    assert dissimilarity.weighted_hamming_distance(data["s1"], data["s2"]) == 3 / 5


def test_weighted_hamming_distance_priors_negative_log(data):
    result = dissimilarity.weighted_hamming_distance(data["s1"], data["s2"], weights=data["nlweights"])
    priors = data["priors"]
    expected = np.sum([-np.log(priors[0][1]), -(np.log(priors[5][2]) + np.log(priors[5][3]))])
    assert result == expected / 5


def test_weighted_hamming_distance_priors_inverse(data):
    result = dissimilarity.weighted_hamming_distance(data["s1"], data["s2"], weights=data["iweights"])
    priors = data["priors"]
    expected = np.sum([1 / priors[0][1], 1 / priors[5][2] + 1 / priors[5][3]])
    assert result == expected / 5


def test_weighted_hamming_distance_priors_sq_inverse(data):
    result = dissimilarity.weighted_hamming_distance(
        data["s1"], data["s2"], weights=data["sqiweights"]
    )
    priors = data["priors"]
    expected = np.sum(
        [np.sqrt(1 / priors[0][1]), np.sqrt(1 / priors[5][2]) + np.sqrt(1 / priors[5][3])]
    )
    assert result == expected / 5


def test_weighted_hamming_distance_all_missing(data):
    assert (
        dissimilarity.weighted_hamming_distance(
            data["s1"], data["all_missing"], weights=data["nlweights"]
        )
        == 0
    )


# ── hamming_similarity_without_missing ────────────────────────────────────────


def test_hamming_similarity_without_missing_identical(data):
    assert dissimilarity.hamming_similarity_without_missing(data["s1"], data["s1"], -1) == 3


def test_hamming_similarity_without_missing_no_priors(data):
    assert dissimilarity.hamming_similarity_without_missing(data["s1"], data["s2"], -1) == 2


def test_hamming_similarity_without_missing_priors(data):
    result = dissimilarity.hamming_similarity_without_missing(
        data["s1"], data["s2"], -1, weights=data["nlweights"]
    )
    priors = data["priors"]
    assert result == np.sum([-np.log(priors[1][1]), -np.log(priors[4][1])])


def test_hamming_similarity_without_missing_all_missing(data):
    assert (
        dissimilarity.hamming_similarity_without_missing(
            data["s1"], data["all_missing"], -1, weights=data["nlweights"]
        )
        == 0
    )


# ── hamming_similarity_normalized_over_missing ────────────────────────────────


def test_hamming_similarity_normalized_identical(data):
    assert (
        dissimilarity.hamming_similarity_normalized_over_missing(data["s1"], data["s1"], -1)
        == 3 / 5
    )


def test_hamming_similarity_normalized_no_priors(data):
    assert (
        dissimilarity.hamming_similarity_normalized_over_missing(data["s1"], data["s2"], -1)
        == 2 / 5
    )


def test_hamming_similarity_normalized_priors(data):
    result = dissimilarity.hamming_similarity_normalized_over_missing(
        data["s1"], data["s2"], -1, weights=data["nlweights"]
    )
    priors = data["priors"]
    assert result == np.sum([-np.log(priors[1][1]), -np.log(priors[4][1])]) / 5


def test_hamming_similarity_normalized_all_missing(data):
    assert (
        dissimilarity.hamming_similarity_normalized_over_missing(
            data["s1"], data["all_missing"], -1, weights=data["nlweights"]
        )
        == 0
    )


# ── weighted_hamming_similarity ───────────────────────────────────────────────


def test_weighted_hamming_similarity_identical(data):
    assert dissimilarity.weighted_hamming_similarity(data["s1"], data["s1"], -1) == 8 / 5


def test_weighted_hamming_similarity_no_priors(data):
    assert dissimilarity.weighted_hamming_similarity(data["s1"], data["s2"], -1) == 1


def test_weighted_hamming_similarity_priors(data):
    result = dissimilarity.weighted_hamming_similarity(
        data["s1"], data["s2"], -1, weights=data["nlweights"]
    )
    priors = data["priors"]
    assert result == np.sum([-np.log(priors[1][1]) * 2, -np.log(priors[4][1]) * 2]) / 5


def test_weighted_hamming_similarity_all_missing(data):
    assert (
        dissimilarity.weighted_hamming_similarity(
            data["s1"], data["all_missing"], -1, weights=data["nlweights"]
        )
        == 0
    )


# ── cluster dissimilarities ───────────────────────────────────────────────────


def test_cluster_dissimilarity(data):
    result = dissimilarity.cluster_dissimilarity(
        dissimilarity.weighted_hamming_distance,
        data["s1"],
        data["ambiguous"],
        -1,
        data["nlweights"],
        np.mean,
    )
    np.testing.assert_almost_equal(result, 1.2544, decimal=4)


def test_cluster_dissimilarity_weighted_hamming_distance_min_linkage(data):
    result = dissimilarity.cluster_dissimilarity_weighted_hamming_distance_min_linkage(
        data["s1"], data["ambiguous_no_missing"], -1, None
    )
    np.testing.assert_almost_equal(result, 0.4, decimal=4)

    result = dissimilarity.cluster_dissimilarity_weighted_hamming_distance_min_linkage(
        data["s1"], data["ambiguous"], -1, None
    )
    np.testing.assert_almost_equal(result, 0.4444, decimal=4)


# ── hamming_distance ──────────────────────────────────────────────────────────


def test_hamming_distance(data):
    assert dissimilarity.hamming_distance(data["s1"], data["s2"]) == 3


def test_hamming_distance_ignore_missing(data):
    assert dissimilarity.hamming_distance(data["s1"], data["s2"], ignore_missing_state=True) == 2
    assert (
        dissimilarity.hamming_distance(data["s1"], data["all_missing"], ignore_missing_state=True)
        == 0
    )


# ── string resolution ─────────────────────────────────────────────────────────


def test_resolve_dissimilarity_by_name():
    fn = dissimilarity._resolve_dissimilarity("weighted_hamming_distance")
    assert fn is dissimilarity.weighted_hamming_distance


def test_resolve_dissimilarity_unknown_raises():
    with pytest.raises(ValueError):
        dissimilarity._resolve_dissimilarity("not_a_metric")
