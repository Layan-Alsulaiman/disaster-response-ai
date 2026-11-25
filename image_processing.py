"""Image preparation and damage-overlay rendering.

Depends only on Pillow and NumPy so it can be tested without PyTorch.

Preprocessing follows the model card of
BiliSakura/xBD-post-bda-models-transformers (the checkpoint ships no
preprocessor_config.json, so no Hugging Face image processor is used):

    RGB -> float in [0, 1] (same as torchvision `to_tensor`)
        -> normalize with ImageNet mean/std
        -> NCHW float32
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- Class definitions (must match the checkpoint's config.json id2label) ---
CLASS_NAMES = {
    0: "Background",
    1: "No Damage",
    2: "Minor Damage",
    3: "Major Damage",
    4: "Destroyed",
}
NUM_CLASSES = len(CLASS_NAMES)

# RGB colours for the overlay. Background is drawn fully transparent.
CLASS_COLORS = {
    0: (0, 0, 0),         # Background (transparent)
    1: (52, 195, 143),    # No Damage   - green   #34C38F
    2: (242, 210, 75),    # Minor       - yellow  #F2D24B
    3: (240, 140, 60),    # Major       - orange  #F08C3C
    4: (229, 72, 77),     # Destroyed   - red     #E5484D
}

# --- Normalization from the model card ---
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# The checkpoint was trained and evaluated on 1024x1024 xBD tiles.
TRAINING_TILE_SIZE = 1024

ImageSource = Union[str, Path, BinaryIO, Image.Image]


def load_rgb_image(source: ImageSource) -> Image.Image:
    """Open an image (path, file-like object or PIL image) and convert it to RGB."""
    if isinstance(source, Image.Image):
        img = source
    else:
        if isinstance(source, (str, Path)) and not Path(source).is_file():
            raise FileNotFoundError(f"Image not found: {source}")
        img = Image.open(source)
        img.load()
    return img.convert("RGB")


def resize_for_inference(image: Image.Image, max_side: int | None) -> Image.Image:
    """Downscale so the longest side is at most `max_side` (never upscales).

    xBD tiles are 1024x1024 and pass through unchanged with the default.
    Larger uploads are reduced to keep CPU inference time and memory
    reasonable; predictions are upsampled back to the original size afterwards.
    """
    if max_side is None:
        return image
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return image.resize(new_size, Image.BILINEAR)


def preprocess(image: Image.Image) -> np.ndarray:
    """Convert an RGB PIL image to a normalized float32 array of shape (1, 3, H, W)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0          # H, W, 3 in [0, 1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD                # ImageNet normalization
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]             # 1, 3, H, W
    return np.ascontiguousarray(arr, dtype=np.float32)


def validate_mask(mask: np.ndarray, expected_hw: tuple[int, int] | None = None) -> None:
    """Raise ValueError if `mask` is not a 2-D class-ID mask with values in 0..4."""
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2-D, got shape {mask.shape}")
    if expected_hw is not None and mask.shape != tuple(expected_hw):
        raise ValueError(f"Mask shape {mask.shape} != expected {tuple(expected_hw)}")
    if mask.size and (mask.min() < 0 or mask.max() >= NUM_CLASSES):
        raise ValueError(
            f"Mask values must be in 0..{NUM_CLASSES - 1}, got {mask.min()}..{mask.max()}"
        )


def class_pixel_counts(mask: np.ndarray) -> dict[int, int]:
    """Number of pixels per predicted class ID (0..4). These are pixels, not buildings."""
    counts = np.bincount(mask.ravel().astype(np.int64), minlength=NUM_CLASSES)
    return {cid: int(counts[cid]) for cid in range(NUM_CLASSES)}


def colorize_mask(mask: np.ndarray, alpha: float = 1.0) -> Image.Image:
    """Render a class-ID mask as an RGBA image. Background pixels are transparent."""
    validate_mask(mask)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for cid, color in CLASS_COLORS.items():
        if cid == 0:
            continue  # leave background fully transparent
        sel = mask == cid
        rgba[sel, :3] = color
        rgba[sel, 3] = round(alpha * 255)
    return Image.fromarray(rgba)  # (H, W, 4) uint8 -> RGBA


def overlay_mask(image: Image.Image, mask: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Blend the coloured damage mask over the satellite image. Returns RGB."""
    base = image.convert("RGBA")
    if (mask.shape[1], mask.shape[0]) != base.size:
        raise ValueError(
            f"Mask size {mask.shape[::-1]} (w, h) does not match image size {base.size}"
        )
    layer = colorize_mask(mask, alpha=alpha)
    return Image.alpha_composite(base, layer).convert("RGB")


# ---------------------------------------------------------------------------
# Priority map (3x3 zone grid drawn over the image)
# ---------------------------------------------------------------------------

def _font(size: int):
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1 (scalable)
    except TypeError:                               # older Pillow
        return ImageFont.load_default()


def draw_zone_grid(
    image: Image.Image,
    zones,
    priority_colors: dict[str, tuple[int, int, int]],
    selected: str | None = None,
    fill_alpha: float = 0.22,
    label=None,
) -> Image.Image:
    """Draw the zone grid over the image: light priority tint, thin boundaries,
    a priority-coloured inner border and a compact label badge per zone.

    `zones` are objects with .bbox (x0, y0, x1, y1), .letter, .priority and
    .score (None for Insufficient Data) - e.g. ranking.Zone. `selected` is a
    zone letter or name to outline. `label(zone) -> str` overrides the badge
    text after the zone letter (default: "<priority> <score>").
    """
    base = image.convert("RGBA")
    w, h = base.size
    unit = min(w, h)
    line = max(1, round(unit / 512))             # grid line width
    edge = max(2, round(unit / 400))             # priority border width
    inset = edge + line

    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    font = _font(max(11, round(unit / 46)))
    pad = max(3, round(unit / 170))

    for z in zones:
        r, g, b = priority_colors[z.priority]
        x0, y0, x1, y1 = z.bbox
        scored = z.score is not None
        d.rectangle((x0, y0, x1 - 1, y1 - 1), fill=(r, g, b, round(255 * fill_alpha * (1 if scored else 0.5))))
        if scored:  # zones without buildings recede: faint tint, no coloured border
            d.rectangle((x0 + inset, y0 + inset, x1 - 1 - inset, y1 - 1 - inset),
                        outline=(r, g, b, 225), width=edge)
        d.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 255, 255, 80), width=line)

        if label is not None:
            sub = label(z)
        else:
            sub = z.priority if z.score is None else f"{z.priority} {z.score:.1f}"
        text = f"{z.letter}  {sub}"
        tx, ty = x0 + inset + edge + pad, y0 + inset + edge + pad
        left, top, right, bottom = d.textbbox((0, 0), text, font=font)
        dot = font.size * 0.55
        bw = (right - left) + dot + pad * 3
        bh = (bottom - top) + pad * 2
        d.rounded_rectangle((tx, ty, tx + bw, ty + bh), radius=pad * 1.5, fill=(15, 20, 32, 205))
        cy = ty + bh / 2
        d.ellipse((tx + pad, cy - dot / 2, tx + pad + dot, cy + dot / 2), fill=(r, g, b, 255))
        d.text((tx + pad * 2 + dot, ty + pad - top), text, font=font, fill=(241, 245, 249, 255))

    if selected:
        sel = next((z for z in zones if selected in (z.letter, getattr(z, "name", None))), None)
        if sel:
            x0, y0, x1, y1 = sel.bbox
            d.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 255, 255, 255), width=max(3, line * 3))

    return Image.alpha_composite(base, layer).convert("RGB")


def crop_box(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    return image.crop(bbox)
