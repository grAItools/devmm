"""Smoke tests for the freshly scaffolded package."""

import devmm


def test_package_imports() -> None:
    assert devmm.__version__
