"""SegFormer-B0 (xBD post-disaster building damage) loading and inference.

Checkpoint: BiliSakura/xBD-post-bda-models-transformers, subfolder
segformer-b0/xbd-baseline/best (CC BY-NC-SA 4.0).

The model takes ONE post-disaster RGB image. It does not perform
before/after change detection; the pre-disaster image is for display only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from image_processing import (
    CLASS_NAMES,
    NUM_CLASSES,
    TRAINING_TILE_SIZE,
    ImageSource,
    load_rgb_image,
    preprocess,
    resize_for_inference,
    validate_mask,
)

try:
    import torch
    import torch.nn.functional as F
    from transformers import SegformerForSemanticSegmentation
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError(
        "model.py needs PyTorch and Hugging Face Transformers. "
        "Install them with: pip install -r requirements.txt"
    ) from exc

MODEL_REPO = "BiliSakura/xBD-post-bda-models-transformers"
MODEL_CHECKPOINT = "segformer-b0/xbd-baseline/best"

# id2label as published in the checkpoint's config.json
EXPECTED_ID2LABEL = {0: "background", 1: "no_damage", 2: "minor", 3: "major", 4: "destroy"}


class ModelLoadError(RuntimeError):
    """Raised when the pretrained checkpoint cannot be downloaded or loaded."""


@dataclass
class Prediction:
    mask: np.ndarray                 # (H, W) uint8 class IDs 0..4, original image size
    original_size: tuple[int, int]   # (width, height) of the input image
    model_input_size: tuple[int, int]  # (width, height) actually fed to the model
    device: str
    inference_seconds: float         # forward pass + upsampling + argmax


def get_device() -> str:
    """CUDA when available, otherwise CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _check_config(model) -> None:
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    if model.config.num_labels != NUM_CLASSES or id2label != EXPECTED_ID2LABEL:
        raise ModelLoadError(
            "Unexpected checkpoint labels: "
            f"num_labels={model.config.num_labels}, id2label={id2label}. "
            f"Expected {EXPECTED_ID2LABEL}."
        )


@lru_cache(maxsize=None)
def _load_model_cached(device: str):
    try:
        model = SegformerForSemanticSegmentation.from_pretrained(
            MODEL_REPO, subfolder=MODEL_CHECKPOINT
        )
    except Exception as exc:  # network, auth, missing files, incompatible version...
        raise ModelLoadError(
            f"Could not load '{MODEL_REPO}' (subfolder '{MODEL_CHECKPOINT}'): {exc}\n"
            "Check your internet connection and that huggingface.co is reachable. "
            "After one successful download the checkpoint is cached locally "
            "(~15 MB) and can be reused offline with HF_HUB_OFFLINE=1."
        ) from exc
    _check_config(model)
    return model.to(device).eval()


def load_model(device: str | None = None):
    """Return the pretrained SegFormer-B0, loaded once per device and cached."""
    return _load_model_cached(device or get_device())


def predict(
    image: ImageSource,
    max_side: int | None = TRAINING_TILE_SIZE,
    device: str | None = None,
    model=None,
) -> Prediction:
    """Run damage segmentation on a post-disaster RGB image.

    Images whose longest side exceeds `max_side` are downscaled for the forward
    pass; logits are always upsampled (bilinear) back to the ORIGINAL image
    size before argmax, so the returned mask aligns pixel-for-pixel with the input.

    `model` is only for tests that inject a small model; normal use leaves it None
    so the cached pretrained checkpoint is used.
    """
    device = device or get_device()
    if model is None:
        model = load_model(device)
    else:
        model = model.to(device).eval()

    img = load_rgb_image(image)
    orig_w, orig_h = img.size
    model_img = resize_for_inference(img, max_side)

    pixel_values = torch.from_numpy(preprocess(model_img)).to(device)

    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(pixel_values=pixel_values).logits      # (1, 5, H/4, W/4)
        logits = F.interpolate(
            logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        mask = logits.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()  # .cpu() syncs CUDA
    elapsed = time.perf_counter() - start

    validate_mask(mask, expected_hw=(orig_h, orig_w))
    return Prediction(
        mask=mask,
        original_size=(orig_w, orig_h),
        model_input_size=model_img.size,
        device=device,
        inference_seconds=elapsed,
    )


__all__ = [
    "CLASS_NAMES",
    "MODEL_CHECKPOINT",
    "MODEL_REPO",
    "ModelLoadError",
    "Prediction",
    "get_device",
    "load_model",
    "predict",
]
