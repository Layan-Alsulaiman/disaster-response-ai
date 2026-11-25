"""Streamlit dashboard smoke tests (headless, via streamlit.testing.AppTest).

    pytest tests/test_app.py                       # dashboard opens without errors
    RUN_MODEL_TESTS=1 pytest tests/test_app.py     # + click "Analyze with AI" on a real sample case

The second test runs the REAL checkpoint on the first case in data/samples/.
"""

import os
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

import data_loader as dl  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_dashboard_opens_without_errors():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("Analyze with AI" in b.label for b in at.button)


@pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 to run the real model")
def test_end_to_end_real_inference_on_sample_case():
    pytest.importorskip("torch")
    if not dl.discover_cases():
        pytest.skip("no sample cases in data/samples/")

    at = AppTest.from_file(APP, default_timeout=600).run()
    analyze = next(b for b in at.button if "Analyze with AI" in b.label)
    at = analyze.click().run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    assert "analysis" in at.session_state
    mask = at.session_state["analysis"]["mask"]
    assert set(mask.ravel().tolist()) <= {0, 1, 2, 3, 4}
    # ranking table + zone buttons rendered from the real prediction
    assert len(at.dataframe) >= 1
    assert sum(1 for b in at.button if b.key and b.key.startswith("zone_")) == 9
