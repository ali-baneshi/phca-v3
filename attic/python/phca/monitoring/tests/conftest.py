"""Shared fixtures for phca.monitoring tests."""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def qt_app():
    from phca.monitoring.qt_dashboard import make_app

    app = make_app()
    yield app
    app.processEvents()
