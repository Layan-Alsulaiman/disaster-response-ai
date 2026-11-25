"""Compare AI predictions with xBD ground truth (Phase 4, kept separate from inference).

Ground-truth building polygons from an xBD post-disaster label JSON are
rasterised into a reference mask (0 background, 1..4 damage, 255 =
un-classified / ignored). The reference mask is used ONLY to compute
metrics; it is never shown or used as an AI prediction.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from image_processing import CLASS_NAMES, NUM_CLASSES

IGNORE_INDEX = 255
SUBTYPE_TO_CLASS = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": IGNORE_INDEX,
}


def rasterize_labels(buildings, size: tuple[int, int]) -> np.ndarray:
    """Rasterise [(polygon_xy, subtype), ...] into an (H, W) uint8 reference mask.

    `size` is (width, height) of the image the polygons belong to.
    """
    img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img)
    for poly, subtype in buildings:
        cls = SUBTYPE_TO_CLASS.get(subtype, IGNORE_INDEX)
        draw.polygon([(float(x), float(y)) for x, y in poly], fill=cls)
    return np.asarray(img, dtype=np.uint8).copy()


def confusion_matrix(gt: np.ndarray, pred: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Rows = ground truth, columns = prediction. Pixels with gt == 255 are ignored."""
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch: gt {gt.shape} vs pred {pred.shape}")
    valid = gt != IGNORE_INDEX
    g = gt[valid].astype(np.int64)
    p = pred[valid].astype(np.int64)
    return np.bincount(g * num_classes + p, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def _div(a, b):
    return float(a) / float(b) if b else float("nan")


def metrics_from_confusion(cm: np.ndarray) -> dict:
    """Per-class IoU / precision / recall plus summary metrics (NaN where undefined)."""
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    per_class = {}
    for c in range(cm.shape[0]):
        per_class[c] = {
            "class": CLASS_NAMES.get(c, str(c)),
            "iou": _div(tp[c], tp[c] + fp[c] + fn[c]),
            "precision": _div(tp[c], tp[c] + fp[c]),
            "recall": _div(tp[c], tp[c] + fn[c]),
            "gt_pixels": int(cm[c].sum()),
            "pred_pixels": int(cm[:, c].sum()),
        }

    # Building localisation: background vs any building class
    bg_bg = cm[0, 0]
    bld_bld = cm[1:, 1:].sum()
    bld_fp = cm[0, 1:].sum()
    bld_fn = cm[1:, 0].sum()
    damage_ious = [per_class[c]["iou"] for c in range(1, cm.shape[0]) if not np.isnan(per_class[c]["iou"])]

    return {
        "per_class": per_class,
        "pixel_accuracy": _div(tp.sum(), cm.sum()),
        "building_iou": _div(bld_bld, bld_bld + bld_fp + bld_fn),
        "mean_damage_iou": float(np.mean(damage_ious)) if damage_ious else float("nan"),
        "valid_pixels": int(cm.sum()),
        "background_correct": int(bg_bg),
    }


def evaluate_prediction(pred_mask: np.ndarray, buildings) -> dict:
    """Metrics for one predicted mask against xBD building polygons."""
    h, w = pred_mask.shape
    gt = rasterize_labels(buildings, (w, h))
    cm = confusion_matrix(gt, pred_mask)
    result = metrics_from_confusion(cm)
    result["confusion_matrix"] = cm
    return result
