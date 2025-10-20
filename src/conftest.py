import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked as slow",
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to skip by default")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # if --runslow given, do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
