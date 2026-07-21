"""The /folders endpoint must flag a granted folder the engine can't READ.

macOS TCC can leave a folder granted in onboarding but unreadable by the launchd
sidecar. Without a signal the folder just shows "0 files" and the app looks
broken. `_root_needs_permission` is the probe that surfaces it.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from find_and_seek.api.server import _root_needs_permission


def test_readable_folder_does_not_need_permission():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "a.txt").write_text("hi")
        assert _root_needs_permission(Path(tmp)) is False


def test_missing_folder_is_not_a_permission_problem():
    # A non-existent path is "exists=False", not "needs_permission".
    assert _root_needs_permission(Path("/no/such/folder/anywhere")) is False


def test_unreadable_folder_needs_permission():
    if os.geteuid() == 0:
        pytest.skip("root bypasses directory permission bits")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "locked"
        d.mkdir()
        (d / "secret.txt").write_text("x")
        os.chmod(d, 0o000)  # deny listing — same shape as a TCC denial (EACCES)
        try:
            assert _root_needs_permission(d) is True
        finally:
            os.chmod(d, 0o755)  # restore so TemporaryDirectory cleanup succeeds
