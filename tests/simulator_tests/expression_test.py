"""Tests for brownian_expression() and trajectory_expression()."""

import numpy as np
import pytest

from cassiopeia.simulator import (
    brownian_expression,
    trajectory_expression,
)

N_GENES = 50

# ---------------------------------------------------------------------------
# brownian_expression
# ---------------------------------------------------------------------------


def test_brownian_shape_and_latent(lineage_tree):
    n_leaves = lineage_tree.n_obs
    out = brownian_expression(lineage_tree, latent_dim=8, n_genes=N_GENES, random_seed=0)
    assert out.X.shape == (n_leaves, N_GENES)
    assert out.obsm["X_latent"].shape == (n_leaves, 8)


def test_brownian_deterministic(lineage_tree):
    a = brownian_expression(lineage_tree, latent_dim=8, n_genes=N_GENES, random_seed=7)
    b = brownian_expression(lineage_tree, latent_dim=8, n_genes=N_GENES, random_seed=7)
    np.testing.assert_array_equal(np.asarray(a.X), np.asarray(b.X))


def test_brownian_gaussian_is_continuous(lineage_tree):
    out = brownian_expression(
        lineage_tree, latent_dim=8, n_genes=N_GENES, distribution="gaussian", random_seed=0
    )
    X = np.asarray(out.X)
    assert np.issubdtype(X.dtype, np.floating)
    # Continuous projection is generally not integer-valued and can be negative.
    assert not np.allclose(X, np.round(X))


@pytest.mark.parametrize("distribution", ["poisson", "negative_binomial"])
def test_brownian_counts(lineage_tree, distribution):
    library_size = 10000
    out = brownian_expression(
        lineage_tree,
        latent_dim=8,
        n_genes=N_GENES,
        distribution=distribution,
        library_size=library_size,
        random_seed=0,
    )
    X = np.asarray(out.X)
    assert np.issubdtype(X.dtype, np.integer)
    assert X.min() >= 0
    # Expected per-cell total counts equals the requested library size.
    assert X.sum(axis=1).mean() == pytest.approx(library_size, rel=0.1)


def test_brownian_layer_added(lineage_tree):
    out = brownian_expression(
        lineage_tree, latent_dim=8, n_genes=N_GENES, layer_added="counts", random_seed=0
    )
    assert "counts" in out.layers
    assert out.layers["counts"].shape == (lineage_tree.n_obs, N_GENES)


def test_brownian_bad_params(lineage_tree):
    from cassiopeia.mixins import DataSimulatorError

    with pytest.raises(DataSimulatorError):
        brownian_expression(lineage_tree, latent_dim=0)
    with pytest.raises(DataSimulatorError):
        brownian_expression(lineage_tree, diffusion=-1.0)
    with pytest.raises(DataSimulatorError):
        brownian_expression(lineage_tree, momentum=1.0)


# ---------------------------------------------------------------------------
# trajectory_expression
# ---------------------------------------------------------------------------


def test_trajectory_shape_and_latent(lineage_tree, trajectory):
    n_leaves = lineage_tree.n_obs
    out = trajectory_expression(lineage_tree, trajectory, n_genes=N_GENES, random_seed=0)
    assert out.X.shape == (n_leaves, N_GENES)
    # latent_dim inferred from the fate tree (5).
    assert out.obsm["X_latent"].shape == (n_leaves, 5)


def test_trajectory_deterministic(lineage_tree, trajectory):
    a = trajectory_expression(lineage_tree, trajectory, n_genes=N_GENES, random_seed=3)
    b = trajectory_expression(lineage_tree, trajectory, n_genes=N_GENES, random_seed=3)
    np.testing.assert_array_equal(np.asarray(a.X), np.asarray(b.X))


@pytest.mark.parametrize("distribution", ["poisson", "negative_binomial"])
def test_trajectory_counts(lineage_tree, trajectory, distribution):
    library_size = 10000
    out = trajectory_expression(
        lineage_tree,
        trajectory,
        n_genes=N_GENES,
        distribution=distribution,
        library_size=library_size,
        random_seed=0,
    )
    X = np.asarray(out.X)
    assert np.issubdtype(X.dtype, np.integer)
    assert X.min() >= 0
    assert X.sum(axis=1).mean() == pytest.approx(library_size, rel=0.1)
