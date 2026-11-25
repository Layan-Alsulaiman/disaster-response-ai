"""Tests for image_processing.py (no PyTorch needed).

Masks here are small hand-made arrays used only to test rendering logic;
they are never presented as model predictions.
"""

import numpy as np
import pytest
from PIL import Image

from image_processing import (
    CLASS_COLORS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    class_pixel_counts,
    colorize_mask,
    load_rgb_image,
    overlay_mask,
    preprocess,
    resize_for_inference,
    validate_mask,
)


def test_preprocess_shape_dtype_and_normalization():
    arr = np.zeros((4, 6, 3), dtype=np.uint8)
    arr[0, 0] = [255, 255, 255]
    arr[1, 2] = [10, 128, 200]
    x = preprocess(Image.fromarray(arr))

    assert x.shape == (1, 3, 4, 6)
    assert x.dtype == np.float32
    # black pixel -> -mean/std ; white -> (1-mean)/std ; arbitrary pixel -> formula
    np.testing.assert_allclose(x[0, :, 3, 5], -IMAGENET_MEAN / IMAGENET_STD, rtol=1e-6)
    np.testing.assert_allclose(x[0, :, 0, 0], (1 - IMAGENET_MEAN) / IMAGENET_STD, rtol=1e-6)
    expected = (np.array([10, 128, 200]) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    np.testing.assert_allclose(x[0, :, 1, 2], expected, rtol=1e-5)


def test_preprocess_channel_order_is_rgb():
    arr = np.zeros((2, 2, 3), dtype=np.uint8)
    arr[..., 0] = 255  # pure red
    x = preprocess(Image.fromarray(arr))
    assert x[0, 0].mean() > 2.0      # R channel high
    assert x[0, 1].mean() < -1.5     # G low
    assert x[0, 2].mean() < -1.5     # B low


def test_load_rgb_image_converts_modes(tmp_path):
    rgba = Image.new("RGBA", (5, 3), (10, 20, 30, 128))
    p = tmp_path / "x.png"
    rgba.save(p)
    img = load_rgb_image(p)
    assert img.mode == "RGB" and img.size == (5, 3)
    assert load_rgb_image(Image.new("L", (2, 2))).mode == "RGB"


def test_load_rgb_image_missing_file():
    with pytest.raises(FileNotFoundError):
        load_rgb_image("does/not/exist.png")


def test_resize_for_inference():
    img = Image.new("RGB", (1024, 1024))
    assert resize_for_inference(img, 1024).size == (1024, 1024)       # xBD tile unchanged
    assert resize_for_inference(Image.new("RGB", (300, 200)), 1024).size == (300, 200)  # no upscaling
    assert resize_for_inference(Image.new("RGB", (2048, 1000)), 1024).size == (1024, 500)
    assert resize_for_inference(Image.new("RGB", (4000, 3000)), None).size == (4000, 3000)


def test_validate_mask():
    validate_mask(np.array([[0, 4], [2, 3]], dtype=np.uint8), expected_hw=(2, 2))
    with pytest.raises(ValueError):
        validate_mask(np.array([[0, 5]], dtype=np.uint8))
    with pytest.raises(ValueError):
        validate_mask(np.zeros((2, 2, 1), dtype=np.uint8))
    with pytest.raises(ValueError):
        validate_mask(np.zeros((2, 3), dtype=np.uint8), expected_hw=(3, 2))


def test_class_pixel_counts():
    mask = np.array([[0, 0, 1], [2, 4, 4]], dtype=np.uint8)
    assert class_pixel_counts(mask) == {0: 2, 1: 1, 2: 1, 3: 0, 4: 2}


def test_colorize_mask_background_transparent_and_colors():
    mask = np.array([[0, 1, 2, 3, 4]], dtype=np.uint8)
    rgba = np.asarray(colorize_mask(mask, alpha=0.5))
    assert rgba.shape == (1, 5, 4)
    assert rgba[0, 0, 3] == 0                         # background transparent
    for cid in range(1, 5):
        assert tuple(rgba[0, cid, :3]) == CLASS_COLORS[cid]
        assert rgba[0, cid, 3] == 128


def test_overlay_mask():
    base = Image.new("RGB", (3, 2), (50, 60, 70))
    mask = np.array([[0, 4, 1], [0, 0, 3]], dtype=np.uint8)

    out = np.asarray(overlay_mask(base, mask, alpha=1.0))
    assert out.shape == (2, 3, 3)
    assert tuple(out[0, 0]) == (50, 60, 70)           # background untouched
    assert tuple(out[0, 1]) == CLASS_COLORS[4]
    assert tuple(out[1, 2]) == CLASS_COLORS[3]

    unchanged = np.asarray(overlay_mask(base, mask, alpha=0.0))
    assert (unchanged == np.asarray(base)).all()

    with pytest.raises(ValueError):
        overlay_mask(base, np.zeros((3, 2), dtype=np.uint8))  # transposed size


def test_draw_zone_grid_tints_and_outlines():
    import ranking as rk
    from image_processing import draw_zone_grid

    mask = np.zeros((90, 90), dtype=np.uint8)
    mask[:30, :30] = 4                       # zone A all destroyed -> High
    res = rk.rank_zones(mask)
    base = Image.new("RGB", (90, 90), (0, 0, 0))
    out = np.asarray(draw_zone_grid(base, res.grid_order(), rk.PRIORITY_COLORS, selected="E"))
    assert out.shape == (90, 90, 3)
    # inside zone A (away from label/border) the High (red) tint dominates
    px = out[8, 22]
    assert px[0] > px[1] and px[0] > px[2]
    # an Insufficient Data zone (I) gets a neutral grey tint
    g = out[63, 63]
    assert abs(int(g[0]) - int(g[2])) < 10 and g[0] > 20
    # selected zone E: white outer ring, dark inner ring
    assert tuple(out[31, 40]) == (255, 255, 255)
    assert tuple(out[35, 40]) == (36, 52, 71)
