import pytest


def pytest_addoption(parser):
    """Register command-line options used for optional test groups."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests",
    )
    parser.addoption(
        "--runspatial",
        action="store_true",
        default=False,
        help="run spatial tests",
    )


def pytest_configure(config):
    """Declare custom markers so pytest does not raise unknown marker errors."""
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "spatial: mark test as spatial to run")


def pytest_collection_modifyitems(config, items):
    """Skip marked tests unless the corresponding command-line flag is set."""
    run_slow = config.getoption("--runslow")
    run_spatial = config.getoption("--runspatial")
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_spatial = pytest.mark.skip(reason="need --runspatial option to run")

    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if "spatial" in item.keywords and not run_spatial:
            item.add_marker(skip_spatial)
