"""launchd agent rendering — pure, no launchctl / no live system changes."""

from __future__ import annotations

import plistlib
from pathlib import Path

from find_and_seek.service import launchd


def test_labels_and_paths():
    assert launchd.label("api") == "com.findandseek.api"
    assert launchd.plist_path("worker").name == "com.findandseek.worker.plist"
    assert launchd.plist_path("worker").parent.name == "LaunchAgents"


def test_plist_dict_shape():
    d = launchd.plist_dict("api")
    assert d["Label"] == "com.findandseek.api"
    assert d["RunAtLoad"] is True
    assert d["KeepAlive"] == {"Crashed": True, "SuccessfulExit": False}
    # One consolidated binary, invoked with the service name as its sub-command.
    assert d["ProgramArguments"][0].endswith("findandseek-sidecar")
    assert d["ProgramArguments"][1] == "api"
    assert "PATH" in d["EnvironmentVariables"]
    assert d["StandardErrorPath"].endswith("api.err.log")


def test_all_services_render_and_serialize():
    for name in launchd.SERVICES:
        d = launchd.plist_dict(name)
        blob = plistlib.dumps(d)            # must be a valid plist
        assert plistlib.loads(blob)["Label"] == f"com.findandseek.{name}"


def test_extra_env_merges():
    d = launchd.plist_dict("worker", {"FINDANDSEEK_BATCH_SIZE": "16"})
    assert d["EnvironmentVariables"]["FINDANDSEEK_BATCH_SIZE"] == "16"
    assert d["EnvironmentVariables"]["FINDANDSEEK_MLX_CACHE_MB"] == "512"  # default kept


def test_write_plist_to_temp(tmp_path, monkeypatch):
    # Redirect the agents dir into a temp dir so we never touch ~/Library.
    monkeypatch.setattr(launchd, "AGENTS_DIR", tmp_path / "LaunchAgents")
    monkeypatch.setattr(launchd, "LOG_DIR", tmp_path / "logs")
    path = launchd.write_plist("api")
    assert path.exists()
    data = plistlib.loads(Path(path).read_bytes())
    assert data["Label"] == "com.findandseek.api"


def test_sidecar_binary_when_frozen(monkeypatch):
    # In a frozen .app, this code IS the sidecar binary, so its path is used
    # directly for the agents' ProgramArguments.
    monkeypatch.setattr(launchd.sys, "frozen", True, raising=False)
    monkeypatch.setattr(launchd.sys, "executable", "/Apps/Find and Seek.app/findandseek-sidecar")
    assert launchd._sidecar_binary() == "/Apps/Find and Seek.app/findandseek-sidecar"
    d = launchd.plist_dict("worker")
    assert d["ProgramArguments"] == ["/Apps/Find and Seek.app/findandseek-sidecar", "worker"]


def test_forwarded_env_is_curated(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("FINDANDSEEK_RERANK_DIR", "/x/reranker")
    monkeypatch.setenv("SOME_UNRELATED_VAR", "leak")
    env = launchd._forwarded_env()
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["FINDANDSEEK_RERANK_DIR"] == "/x/reranker"
    assert "SOME_UNRELATED_VAR" not in env
