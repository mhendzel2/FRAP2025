"""Lightweight import tests for package wrappers/proxies."""

from __future__ import annotations

import pytest


def test_core_global_fitting_proxy_importable():
    from frap2025.core import frap_global_fitting

    exported = dir(frap_global_fitting)
    assert isinstance(exported, list)
    assert len(exported) > 0

    # Exercise proxy path for a real exported attribute when present.
    if "LMFIT_AVAILABLE" in exported:
        _ = frap_global_fitting.LMFIT_AVAILABLE

    with pytest.raises(AttributeError):
        getattr(frap_global_fitting, "__not_a_real_attr__")
