# Define custom marker.
#   https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option


import pytest

marker_str = "examples"
option = f"--{marker_str}"


def pytest_addoption(parser):
    parser.addoption(
        option,
        action="store_true",
        default=False,
        help="run example scripts",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", f"{marker_str}: example scripts")


def pytest_collection_modifyitems(config, items):
    if config.getoption(option):
        return
    skip_marker = pytest.mark.skip(reason=f"need {option} option to run")
    for item in items:
        if marker_str in item.keywords:
            item.add_marker(skip_marker)
