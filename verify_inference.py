"""Phase 1 check: run the real SegFormer-B0 checkpoint on one post-disaster image.

Usage:
    python verify_inference.py data/samples/<case>/post.png
    python verify_inference.py path/to/post.png --out outputs --alpha 0.5

Saves <name>_mask.png (class IDs 0..4), <name>_mask_color.png,
<name>_overlay.png and <name>_priority_map.png, and prints per-class pixel
statistics and the 3x3 zone ranking. Exits non-zero if
any check fails.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from image_processing import (
    CLASS_NAMES,
    NUM_CLASSES,
    class_pixel_counts,
    colorize_mask,
    draw_zone_grid,
    load_rgb_image,
    overlay_mask,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="Post-disaster RGB image (e.g. an xBD *_post_disaster.png)")
    parser.add_argument("--out", default="outputs", help="Output directory (default: outputs)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay opacity 0..1")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    args = parser.parse_args()

    try:
        image = load_rgb_image(args.image)
    except (FileNotFoundError, OSError) as exc:
        print(f"Cannot read image: {exc}", file=sys.stderr)
        return 2

    import model as seg  # imported here so a missing torch gives a clear message

    device = args.device or seg.get_device()
    print(f"Device: {device}")

    t0 = time.perf_counter()
    try:
        net = seg.load_model(device)
    except seg.ModelLoadError as exc:
        print(f"MODEL LOAD FAILED:\n{exc}", file=sys.stderr)
        return 2
    print(f"Loaded {seg.MODEL_REPO}/{seg.MODEL_CHECKPOINT} in {time.perf_counter() - t0:.1f}s")
    print(f"  id2label: {net.config.id2label}")

    t1 = time.perf_counter()
    assert seg.load_model(device) is net, "model was not cached"
    print(f"Second load_model() call returned the cached instance ({(time.perf_counter() - t1) * 1000:.2f} ms)")

    pred = seg.predict(image, device=device)
    mask = pred.mask

    # --- checks ---
    w, h = image.size
    checks = {
        "mask is 2-D uint8": mask.ndim == 2 and mask.dtype == np.uint8,
        f"mask size == image size ({h}x{w})": mask.shape == (h, w),
        "class IDs within 0..4": int(mask.min()) >= 0 and int(mask.max()) < NUM_CLASSES,
    }

    counts = class_pixel_counts(mask)
    total = mask.size
    building = total - counts[0]
    print(f"\nImage {w}x{h}, model input {pred.model_input_size[0]}x{pred.model_input_size[1]}, "
          f"inference {pred.inference_seconds:.2f}s on {pred.device}")
    print("Predicted pixels per class (pixels, NOT building counts):")
    for cid in range(NUM_CLASSES):
        share_all = 100 * counts[cid] / total
        share_bld = f"{100 * counts[cid] / building:5.1f}% of building px" if cid and building else ""
        print(f"  {cid} {CLASS_NAMES[cid]:<13} {counts[cid]:>9}  {share_all:5.1f}% of image  {share_bld}")
    print(f"Classes present: {sorted(int(c) for c in np.unique(mask))}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    Image.fromarray(mask).save(out / f"{stem}_mask.png")  # uint8 (H, W) -> "L", values 0..4
    colorize_mask(mask).save(out / f"{stem}_mask_color.png")
    overlay_mask(image, mask, alpha=args.alpha).save(out / f"{stem}_overlay.png")
    print(f"\nSaved mask, colour mask and overlay to {out}/")

    import ranking as rk

    result = rk.rank_zones(mask)
    print("\nZone ranking (3x3 image sectors, default thresholds):")
    for row in rk.zones_table(result):
        score = "  -  " if row["Score"] is None else f"{row['Score']:.2f}"
        print(f"  {str(row['Rank']):>2}  {row['Zone']}  score {score}  {row['Priority']}")
    print("\n" + rk.summarize(result))
    draw_zone_grid(image, result.grid_order(), rk.PRIORITY_COLORS).save(out / f"{stem}_priority_map.png")

    print("\nChecks:")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if building == 0:
        print("  [NOTE] No building pixels predicted - check that this is a post-disaster "
              "satellite tile at roughly xBD resolution.")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
