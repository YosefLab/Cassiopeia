"""Shared fixtures for the preprocessing tests."""

import importlib.util

import matplotlib
import pytest

matplotlib.use("Agg")

from cassiopeia.preprocess._simulate import simulate

HAS_NOVA = importlib.util.find_spec("nova_clone") is not None
requires_nova = pytest.mark.skipif(not HAS_NOVA, reason="nova_clone is not installed")


@pytest.fixture(scope="session")
def sim_factory(tmp_path_factory):
    """Simulate (and cache) a dataset per (construct, scenario)."""
    cache = {}

    def make(construct="weissman", scenario="multi", **kwargs):
        key = (construct, scenario, tuple(sorted(kwargs.items())))
        if key not in cache:
            params = {
                "multi": {"n_clones": 3, "cells_per_clone": 25, "intbcs_per_clone": 5},
                "single": {"n_clones": 1, "cells_per_clone": 40, "intbcs_per_clone": 6},
                "clean": {
                    "n_clones": 2,
                    "cells_per_clone": 15,
                    "intbcs_per_clone": 4,
                    "ambient_rate": 0.0,
                    "doublet_rate": 0.0,
                    "seq_error": 0.0,
                    "cb_error": 0.0,
                    "umi_error": 0.0,
                },
            }[scenario]
            if construct == "gestalt":
                params["intbcs_per_clone"] = 1
            params.update(kwargs)
            d = tmp_path_factory.mktemp(f"sim_{construct}_{scenario}")
            cache[key] = simulate(d, construct, seed=7, **params)
        return cache[key]

    return make
