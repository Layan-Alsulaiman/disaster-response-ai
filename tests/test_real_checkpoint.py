"""Integration test with the REAL pretrained checkpoint and a REAL post-disaster image.

Opt-in because it downloads ~15 MB from Hugging Face:

    RUN_MODEL_TESTS=1 pytest tests/test_real_checkpoint.py -s
    RUN_MODEL_TESTS=1 SAMPLE_POST_IMAGE=path/to/post.png pytest tests/test_real_checkpoint.py -s

Without SAMPLE_POST_IMAGE, the first image under data/samples/ whose name
contains "post" is used.
"""

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="set RUN_MODEL_TESTS=1 to download and run the real checkpoint",
)

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import model as seg  # noqa: E402
from image_processing import load_rgb_image, overlay_mask  # noqa: E402

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


def find_post_image():
    env = os.environ.get("SAMPLE_POST_IMAGE")
    if env:
        return Path(env)
    for ext in ("png", "jpg", "jpeg", "tif", "tiff"):
        hits = sorted(p for p in SAMPLES.rglob(f"*.{ext}") if "post" in p.name.lower())
        if hits:
            return hits[0]
    return None


@pytest.fixture(scope="module")
def net():
    return seg.load_model()


def test_real_checkpoint_loads_and_is_cached(net):
    assert type(net).__name__ == "SegformerForSemanticSegmentation"
    assert net.config.num_labels == 5
    assert seg.load_model() is net
    assert next(net.parameters()).device.type == seg.get_device()


def test_real_inference_on_post_disaster_image(net, tmp_path):
    path = find_post_image()
    if path is None or not path.is_file():
        pytest.skip("no post-disaster image found; set SAMPLE_POST_IMAGE or add one to data/samples/")

    image = load_rgb_image(path)
    pred = seg.predict(image)
    w, h = image.size

    assert pred.mask.shape == (h, w)
    assert pred.mask.dtype == np.uint8
    assert set(np.unique(pred.mask)).issubset({0, 1, 2, 3, 4})

    out = overlay_mask(image, pred.mask, alpha=0.5)
    assert out.size == image.size
    out.save(tmp_path / "overlay.png")
    print(f"\n{path.name}: classes {sorted(np.unique(pred.mask).tolist())}, "
          f"{pred.inference_seconds:.2f}s on {pred.device}")
