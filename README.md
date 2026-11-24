# disaster-response-ai

Hackathon MVP: AI building-damage segmentation on post-disaster satellite imagery, followed by 3×3 zone priority ranking (see `doc/project-spec`).

**Status: Phase 1 (AI integration).** The dashboard is not built yet.

## Model

- Hugging Face repo: [`BiliSakura/xBD-post-bda-models-transformers`](https://huggingface.co/BiliSakura/xBD-post-bda-models-transformers), subfolder `segformer-b0/xbd-baseline/best`
- SegFormer (MiT-B0), fine-tuned on xBD post-disaster images; reported validation micro IoU 40.6 %
- Input: **one post-disaster RGB image**. It does **not** do before/after change detection.
- Output classes: 0 Background, 1 No Damage, 2 Minor, 3 Major, 4 Destroyed
- License: CC BY-NC-SA 4.0 (non-commercial)

Preprocessing (from the model card; the checkpoint has no `preprocessor_config.json`): RGB → `[0, 1]` → ImageNet mean/std normalization. Logits come out at 1/4 resolution and are bilinearly upsampled to the original image size before argmax. Images whose long side exceeds 1024 px (the xBD training tile size) are downscaled for the forward pass only; the returned mask always matches the original image size.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The checkpoint (~15 MB) downloads automatically on first use and is cached by Hugging Face (`~/.cache/huggingface`). After that you can run offline with `HF_HUB_OFFLINE=1`. CUDA is used automatically when available, otherwise CPU.

## Phase 1: verify real inference

Add a real xBD post-disaster tile (see `data/samples/README.md`), then:

```bash
python verify_inference.py data/samples/<case>/<case>_post_disaster.png
```

This loads the checkpoint once, confirms it is cached, runs inference, checks that the mask matches the image size and only contains class IDs 0–4, prints per-class **pixel** statistics, and writes `outputs/<name>_mask.png`, `_mask_color.png` and `_overlay.png`.

From Python:

```python
from model import predict
from image_processing import overlay_mask, load_rgb_image

img = load_rgb_image("post.png")
pred = predict(img)                      # pred.mask: (H, W) uint8, values 0..4
overlay_mask(img, pred.mask, alpha=0.5).save("overlay.png")
```

## Tests

```bash
pytest                                     # offline unit tests
RUN_MODEL_TESTS=1 pytest tests/test_real_checkpoint.py -s   # downloads and runs the real checkpoint
```

`tests/test_model.py` uses a tiny randomly initialised SegFormer only to test shapes, resizing and caching offline; its output is never used as a prediction.

## Notes

- Mask values are predicted damage **pixels**, not building counts.
- The 3×3 zones (later phases) are image sectors, not geographic coordinates.
- This is a decision-support demo, not an operational emergency-response system.
