"""
Pytest configuration for RecommendationFiles tests.
This is so we skip the integration tests if for some reason the data is not available.
"""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests that require real data files (skip with '-m \"not integration\"')"
    )


def pytest_collection_modifyitems(config, items):
    """Add skip marker to integration tests if data files not available."""
    if config.getoption("-m") == "integration":
        return

    pass
