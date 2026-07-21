"""Signal-derived confidence: the model's self-report is capped by extraction volume."""

from find_and_seek.organize.classify import cap_confidence, signal_ceiling


def test_ceiling_tracks_extraction_volume():
    assert signal_ceiling(0) == "low"
    assert signal_ceiling(119) == "low"
    assert signal_ceiling(120) == "medium"
    assert signal_ceiling(499) == "medium"
    assert signal_ceiling(500) == "high"


def test_cap_only_lowers_never_raises():
    # A confident claim on a thin body is capped…
    assert cap_confidence("high", 60) == "low"
    assert cap_confidence("high", 300) == "medium"
    # …a rich body lets the claim stand…
    assert cap_confidence("high", 5000) == "high"
    # …and an already-humble claim is never inflated.
    assert cap_confidence("low", 5000) == "low"
    assert cap_confidence("none", 5000) == "none"
