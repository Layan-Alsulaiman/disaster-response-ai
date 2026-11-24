"""Phase 4: evaluate the real model on local xBD cases that include label JSON.

Usage:
    python evaluate.py                    # all cases in data/samples with labels
    python evaluate.py --root other/dir

For each case: runs real inference on the post-disaster image, rasterises the
xBD ground-truth polygons, and prints IoU / precision / recall per class, the
confusion matrix and inference time. Ground truth is used only for scoring.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from data_loader import SAMPLES_DIR, discover_cases, load_json, xbd_buildings
from evaluation import evaluate_prediction
from image_processing import CLASS_NAMES


def fmt(x: float) -> str:
    return "  n/a" if math.isnan(x) else f"{100 * x:5.1f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(SAMPLES_DIR))
    args = ap.parse_args()

    cases = [c for c in discover_cases(args.root) if c.has_labels]
    if not cases:
        print(f"No cases with *_post_disaster.json labels found in {args.root}. See data/samples/README.md.")
        return 1

    import model as seg

    total_cm = None
    for case in cases:
        pred = seg.predict(case.post_path)
        res = evaluate_prediction(pred.mask, xbd_buildings(load_json(case.labels_path)))
        cm = res["confusion_matrix"]
        total_cm = cm if total_cm is None else total_cm + cm

        print(f"\n=== {case.name} ({case.disaster_type}) — {case.post_path.name}")
        print(f"inference {pred.inference_seconds:.2f}s on {pred.device}; "
              f"building IoU {fmt(res['building_iou'])}%, mean damage IoU {fmt(res['mean_damage_iou'])}%, "
              f"pixel accuracy {fmt(res['pixel_accuracy'])}%")
        print(f"{'class':<14}{'IoU%':>7}{'Prec%':>7}{'Rec%':>7}{'GT px':>10}{'Pred px':>10}")
        for c, m in res["per_class"].items():
            print(f"{m['class']:<14}{fmt(m['iou']):>7}{fmt(m['precision']):>7}{fmt(m['recall']):>7}"
                  f"{m['gt_pixels']:>10}{m['pred_pixels']:>10}")
        print("confusion matrix (rows = ground truth, cols = prediction):")
        print(np.array2string(cm, max_line_width=200))

    if len(cases) > 1:
        from evaluation import metrics_from_confusion
        agg = metrics_from_confusion(total_cm)
        print(f"\n=== All {len(cases)} cases: building IoU {fmt(agg['building_iou'])}%, "
              f"mean damage IoU {fmt(agg['mean_damage_iou'])}%")
        for c, m in agg["per_class"].items():
            print(f"  {CLASS_NAMES[c]:<14} IoU {fmt(m['iou'])}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
