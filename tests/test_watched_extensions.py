"""v1 indexes documents only; code/config is a deferred opt-in (v1.5).

A documents product must not flood the index with a dev workspace's source files
(.js/.ts/.json was ~38% of the live index — pure search noise). Code indexing is
gated behind FINDANDSEEK_INDEX_CODE (the v1.5 user toggle).
"""

from __future__ import annotations

import importlib

import find_and_seek.ingest.extract.router as router


def _reload(monkeypatch, *, index_code: bool):
    if index_code:
        monkeypatch.setenv("FINDANDSEEK_INDEX_CODE", "1")
    else:
        monkeypatch.delenv("FINDANDSEEK_INDEX_CODE", raising=False)
    return importlib.reload(router)


def test_documents_indexed_by_default(monkeypatch):
    r = _reload(monkeypatch, index_code=False)
    for ext in (".pdf", ".docx", ".md", ".txt", ".xlsx", ".eml", ".jpg"):
        assert ext in r.WATCHED_EXTENSIONS, f"{ext} should be a default document type"


def test_code_excluded_by_default(monkeypatch):
    r = _reload(monkeypatch, index_code=False)
    for ext in (".js", ".ts", ".tsx", ".py", ".json", ".xml", ".yml", ".go"):
        assert ext not in r.WATCHED_EXTENSIONS, f"{ext} must NOT be watched in v1"


def test_code_opt_in_unions_code_extensions(monkeypatch):
    r = _reload(monkeypatch, index_code=True)
    assert r.CODE_EXTENSIONS <= r.WATCHED_EXTENSIONS
    assert ".js" in r.WATCHED_EXTENSIONS and ".py" in r.WATCHED_EXTENSIONS
    # Documents are still in regardless.
    assert ".pdf" in r.WATCHED_EXTENSIONS


def test_every_watched_extension_has_an_extractor(monkeypatch):
    # True in both modes — code extractors stay registered so the v1.5 flip works.
    for index_code in (False, True):
        r = _reload(monkeypatch, index_code=index_code)
        assert r.WATCHED_EXTENSIONS <= set(r.EXTRACTORS)


def test_reload_restores_default(monkeypatch):
    # Leave the module in its default state for any later importers in the session.
    _reload(monkeypatch, index_code=False)
    assert ".js" not in router.WATCHED_EXTENSIONS
