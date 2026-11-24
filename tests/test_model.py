"""Offline tests of the inference pipeline in model.py.

A tiny, randomly initialised SegFormer with the same 5 labels is used to check
shapes, resizing, caching and error handling without downloading anything.
Its outputs are meaningless and are never used as damage predictions; the real
checkpoint is exercised in test_real_checkpoint.py.
"""

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import model as seg  # noqa: E402


def tiny_model(id2label=None):
    id2label = id2label or seg.EXPECTED_ID2LABEL
    cfg = transformers.SegformerConfig(
        num_labels=len(id2label),
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
        hidden_sizes=[8, 16, 16, 16],
        depths=[1, 1, 1, 1],
        num_attention_heads=[1, 1, 1, 1],
        decoder_hidden_size=16,
    )
    torch.manual_seed(0)
    return transformers.SegformerForSemanticSegmentation(cfg).eval()


def random_image(w, h):
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


def test_get_device():
    assert seg.get_device() == ("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("size", [(64, 64), (130, 97), (33, 70)])
def test_predict_mask_matches_input_size(size):
    w, h = size
    pred = seg.predict(random_image(w, h), device="cpu", model=tiny_model())
    assert pred.mask.shape == (h, w)
    assert pred.mask.dtype == np.uint8
    assert pred.mask.min() >= 0 and pred.mask.max() <= 4
    assert pred.original_size == (w, h)
    assert pred.model_input_size == (w, h)


def test_predict_downscales_large_input_but_returns_full_size_mask():
    pred = seg.predict(random_image(200, 150), max_side=64, device="cpu", model=tiny_model())
    assert pred.model_input_size == (64, 48)
    assert pred.mask.shape == (150, 200)


def test_predict_matches_manual_reference_pipeline():
    """predict() == the model card's reference code (to_tensor, normalize, upsample, argmax)."""
    net = tiny_model()
    img = random_image(48, 40)
    pred = seg.predict(img, device="cpu", model=net)

    x = torch.tensor(np.asarray(img), dtype=torch.float32).permute(2, 0, 1)[None] / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    with torch.no_grad():
        logits = net(pixel_values=(x - mean) / std).logits
    logits = torch.nn.functional.interpolate(logits, size=(40, 48), mode="bilinear", align_corners=False)
    ref = logits.argmax(1)[0].numpy().astype(np.uint8)
    np.testing.assert_array_equal(pred.mask, ref)


def test_load_model_is_cached(monkeypatch):
    calls = []

    def fake_from_pretrained(repo, subfolder=None, **kw):
        calls.append((repo, subfolder))
        return tiny_model()

    seg._load_model_cached.cache_clear()
    monkeypatch.setattr(seg.SegformerForSemanticSegmentation, "from_pretrained", fake_from_pretrained)
    try:
        a = seg.load_model("cpu")
        b = seg.load_model("cpu")
        assert a is b
        assert calls == [(seg.MODEL_REPO, seg.MODEL_CHECKPOINT)]
    finally:
        seg._load_model_cached.cache_clear()


def test_load_failure_raises_clear_error(monkeypatch):
    def boom(*a, **k):
        raise OSError("connection refused")

    seg._load_model_cached.cache_clear()
    monkeypatch.setattr(seg.SegformerForSemanticSegmentation, "from_pretrained", boom)
    try:
        with pytest.raises(seg.ModelLoadError, match="huggingface.co"):
            seg.load_model("cpu")
    finally:
        seg._load_model_cached.cache_clear()


def test_wrong_label_config_rejected(monkeypatch):
    seg._load_model_cached.cache_clear()
    monkeypatch.setattr(
        seg.SegformerForSemanticSegmentation, "from_pretrained",
        lambda *a, **k: tiny_model({0: "background", 1: "building"}),
    )
    try:
        with pytest.raises(seg.ModelLoadError, match="Unexpected checkpoint labels"):
            seg.load_model("cpu")
    finally:
        seg._load_model_cached.cache_clear()
