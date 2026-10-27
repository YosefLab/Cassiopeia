"""Shared fixtures for the preprocessing tests."""

import importlib.util

import matplotlib
import pytest

matplotlib.use("Agg")

from cassiopeia.preprocess._simulate import simulate


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests (published data)"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow tests using downloaded published datasets")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="needs --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


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
