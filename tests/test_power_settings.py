"""The user-facing performance mode (smooth / full), persisted to power.json.

Smooth is the default and the safe fallback; the app's "Full power" toggle flips
it to full. Env var overrides the file (power users / tests).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from find_and_seek.config import power_settings as ps


@pytest.fixture
def power_file(tmp_path, monkeypatch):
    p = tmp_path / "power.json"
    monkeypatch.setenv("FINDANDSEEK_POWER_SETTINGS", str(p))
    monkeypatch.delenv("FINDANDSEEK_PERFORMANCE_MODE", raising=False)
    return p


def test_default_is_smooth(power_file):
    assert ps.get_mode() == "smooth"
    assert ps.get_settings() == {"mode": "smooth"}


def test_set_full_round_trips(power_file):
    assert ps.set_mode("full") == {"mode": "full"}
    assert ps.get_mode() == "full"
    assert json.loads(power_file.read_text())["mode"] == "full"


def test_unknown_mode_clamps_to_smooth(power_file):
    assert ps.set_mode("ludicrous") == {"mode": "smooth"}
    assert ps.get_mode() == "smooth"


def test_corrupt_file_falls_back_to_smooth(power_file):
    power_file.write_text("{ not json")
    assert ps.get_mode() == "smooth"


def test_env_var_overrides_file(power_file, monkeypatch):
    ps.set_mode("smooth")
    monkeypatch.setenv("FINDANDSEEK_PERFORMANCE_MODE", "full")
    assert ps.get_mode() == "full"
    # An unrecognised env value falls back to the safe default, not the file.
    monkeypatch.setenv("FINDANDSEEK_PERFORMANCE_MODE", "bogus")
    assert ps.get_mode() == "smooth"
