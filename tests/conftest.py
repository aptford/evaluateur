"""Shared test configuration.

Load .env early so that ``skipif`` markers evaluated at collection time
can see API-key environment variables.
"""

import pytest
from dotenv import load_dotenv

load_dotenv()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--include-env",
        action="store_true",
        default=False,
        help="Run tests that require environment credentials (API keys).",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "env: marks tests that require environment credentials (skipped unless --include-env is passed)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--include-env"):
        return
    skip = pytest.mark.skip(reason="Need --include-env to run")
    for item in items:
        if "env" in item.keywords:
            item.add_marker(skip)
