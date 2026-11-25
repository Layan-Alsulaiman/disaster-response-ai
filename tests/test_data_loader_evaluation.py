"""Tests for data_loader.py and evaluation.py with small synthetic files in xBD format."""

import json

import numpy as np
import pytest
from PIL import Image

import data_loader as dl
import evaluation as ev


def xbd_labels(buildings, disaster="hurricane-michael", dtype="wind"):
    return {
        "metadata": {"disaster": disaster, "disaster_type": dtype, "width": 20, "height": 20,
                     "capture_date": "2018-10-13T16:56:46.000Z"},
        "features": {
            "lng_lat": [{"wkt": "POLYGON ((-85.0 30.0, -85.0 30.2, -84.8 30.2, -85.0 30.0))"}],
            "xy": [
                {"properties": {"feature_type": "building", "subtype": st, "uid": str(i)},
                 "wkt": "POLYGON ((" + ", ".join(f"{x} {y}" for x, y in poly) + "))"}
                for i, (poly, st) in enumerate(buildings)
            ],
        },
    }


def write_case(folder, prefix="hurricane-michael_00000123", labels=None, case_json=None, pre=True):
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20)).save(folder / f"{prefix}_post_disaster.png")
    if pre:
        Image.new("RGB", (20, 20)).save(folder / f"{prefix}_pre_disaster.png")
    if labels is not None:
        (folder / f"{prefix}_post_disaster.json").write_text(json.dumps(labels))
    if case_json is not None:
        (folder / "case.json").write_text(json.dumps(case_json))


# --- data_loader ------------------------------------------------------------

def test_discover_case_with_xbd_metadata(tmp_path):
    write_case(tmp_path / "michael", labels=xbd_labels([]))
    (case,) = dl.discover_cases(tmp_path)
    assert case.post_path.name.endswith("_post_disaster.png")
    assert case.pre_path.name.endswith("_pre_disaster.png")
    assert case.has_labels
    assert case.disaster_type == "Hurricane / Wind"
    assert case.name.startswith("Hurricane Michael")
    assert case.capture_date == "2018-10-13"


def test_case_json_overrides_metadata(tmp_path):
    write_case(tmp_path / "c1", labels=xbd_labels([]), case_json={"name": "Demo", "disaster_type": "Flood"})
    (case,) = dl.discover_cases(tmp_path)
    assert (case.name, case.disaster_type) == ("Demo", "Flood")


def test_case_without_pre_or_labels(tmp_path):
    write_case(tmp_path / "solo", prefix="x", pre=False)
    (case,) = dl.discover_cases(tmp_path)
    assert case.pre_path is None and not case.has_labels and case.disaster_type == "Unknown"


def test_folder_without_post_image_is_ignored(tmp_path):
    (tmp_path / "empty").mkdir()
    Image.new("RGB", (4, 4)).save(tmp_path / "empty" / "something_pre_disaster.png")
    assert dl.discover_cases(tmp_path) == []


def test_loose_xbd_pairs_grouped(tmp_path):
    write_case(tmp_path, prefix="santa-rosa-wildfire_00000001", labels=xbd_labels([], "santa-rosa-wildfire", "fire"))
    (case,) = dl.discover_cases(tmp_path)
    assert case.case_id == "santa-rosa-wildfire_00000001"
    assert case.disaster_type == "Wildfire" and case.pre_path is not None


def test_missing_root_returns_empty(tmp_path):
    assert dl.discover_cases(tmp_path / "nope") == []


def test_prediction_word_not_mistaken_for_pre(tmp_path):
    folder = tmp_path / "c"
    write_case(folder, prefix="a", pre=False)
    Image.new("RGB", (4, 4)).save(folder / "prediction_overlay.png")
    (case,) = dl.discover_cases(tmp_path)
    assert case.pre_path is None


def test_wkt_parsing_and_location():
    assert dl.parse_wkt_polygon("POLYGON ((1 2, 3.5 4, 5 6, 1 2))") == [(1, 2), (3.5, 4), (5, 6), (1, 2)]
    assert dl.parse_wkt_polygon("garbage") == []
    lat, lon = dl.tile_location(xbd_labels([]))
    assert lat == pytest.approx(30.1) and lon == pytest.approx(-84.95)
    assert dl.tile_location({"features": {}}) is None


# --- evaluation -------------------------------------------------------------

SQUARE = [(2, 2), (7, 2), (7, 7), (2, 7)]
SQUARE2 = [(12, 12), (17, 12), (17, 17), (12, 17)]


def test_rasterize_labels_classes_and_ignore():
    labels = xbd_labels([(SQUARE, "destroyed"), (SQUARE2, "un-classified")])
    gt = ev.rasterize_labels(dl.xbd_buildings(labels), (20, 20))
    assert gt.shape == (20, 20)
    assert gt[4, 4] == 4 and gt[14, 14] == ev.IGNORE_INDEX and gt[0, 0] == 0


def test_perfect_prediction_metrics():
    gt = ev.rasterize_labels([(SQUARE, "major-damage")], (20, 20))
    res = ev.evaluate_prediction(gt.copy(), [(SQUARE, "major-damage")])
    assert res["pixel_accuracy"] == 1.0
    assert res["building_iou"] == 1.0
    assert res["per_class"][3]["iou"] == 1.0
    assert np.isnan(res["per_class"][4]["iou"])      # class absent in both -> undefined


def test_confusion_and_partial_metrics():
    gt = np.array([[0, 1, 1, 255]], dtype=np.uint8)
    pred = np.array([[1, 1, 2, 4]], dtype=np.uint8)
    cm = ev.confusion_matrix(gt, pred)
    assert cm.sum() == 3                               # ignored pixel dropped
    assert cm[0, 1] == 1 and cm[1, 1] == 1 and cm[1, 2] == 1
    m = ev.metrics_from_confusion(cm)
    assert m["per_class"][1]["iou"] == pytest.approx(1 / 3)
    assert m["per_class"][1]["precision"] == pytest.approx(0.5)
    assert m["per_class"][1]["recall"] == pytest.approx(0.5)
    assert m["building_iou"] == pytest.approx(2 / 3)


def test_confusion_shape_mismatch():
    with pytest.raises(ValueError):
        ev.confusion_matrix(np.zeros((2, 2), np.uint8), np.zeros((2, 3), np.uint8))
