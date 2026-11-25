"""Streamlit dashboard smoke tests (headless, via streamlit.testing.AppTest).

    pytest tests/test_app.py                       # the app opens on New Assessment without errors
    RUN_MODEL_TESTS=1 pytest tests/test_app.py     # + run the REAL model on a sample case and visit every view
"""

import os
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

import data_loader as dl  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_opens_on_new_assessment():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert at.session_state["view"] == "new"
    assert any(b.label == "Analyze with AI" for b in at.button)
    # result views are unavailable until an assessment exists
    nav = {b.key: b for b in at.button if b.key and b.key.startswith("nav_")}
    assert nav["nav_overview"].disabled and nav["nav_detail"].disabled


@pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 to run the real model")
def test_end_to_end_real_inference_all_views():
    pytest.importorskip("torch")
    if not dl.discover_cases():
        pytest.skip("no sample cases in data/samples/")

    at = AppTest.from_file(APP, default_timeout=600).run()
    at = next(b for b in at.button if b.label == "Analyze with AI").click().run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]

    # Response Overview is shown after a successful assessment
    assert at.session_state["view"] == "overview"
    a = at.session_state["assessment"]
    assert set(a["mask"].ravel().tolist()) <= {0, 1, 2, 3, 4}
    zone_buttons = [b for b in at.button if b.key and b.key.startswith("zbtn_")]
    if zone_buttons:  # selecting a zone from the ranked list
        at = zone_buttons[-1].click().run()
        assert at.session_state["zone"] == zone_buttons[-1].key.split("_")[1]

    # Detailed Analysis
    at = next(b for b in at.button if b.key == "nav_detail").click().run()
    assert not at.exception, at.exception
    assert at.session_state["view"] == "detail"

    # back to New Assessment keeps the result available
    at = next(b for b in at.button if b.key == "nav_new").click().run()
    assert not at.exception and at.session_state["assessment"] is not None
