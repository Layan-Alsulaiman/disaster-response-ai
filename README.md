# 🛰️ Disaster Damage Intelligence

AI-powered building-damage assessment on post-disaster satellite imagery, turned into ranked response zones.
Hackathon MVP. Full specification: [`doc/project-spec`](doc/project-spec).

A pretrained SegFormer-B0 model (fine-tuned on xBD) segments every pixel of a **post-disaster** image into
Background / No Damage / Minor / Major / Destroyed. The app splits the image into a 3 × 3 grid, scores each zone
from the predicted pixels, and ranks the zones as High / Medium / Low priority or Insufficient Data, with an
explanation for each. Everything is shown in a Streamlit dashboard.

> Decision-support demonstration, not an operational emergency-response system. It ranks **visual damage
> severity** only. It does not estimate casualties, population exposure or rescue urgency.

## Pipeline

```
post-disaster image ─► preprocessing ─► SegFormer-B0 (xBD) ─► 5-class mask ─► 3×3 zones ─► scores ─► priorities ─► dashboard
pre-disaster image  ─────────────────────────────── shown for visual comparison only ──────────────────────────► dashboard
```

The model uses one image. It does **not** perform before/after change detection.

## Quick start

Requires Python 3.11+ (3.10–3.13 should work).

```bash
git clone https://github.com/Layan-Alsulaiman/disaster-response-ai.git
cd disaster-response-ai
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

- **CPU-only machines:** to avoid the large CUDA build of PyTorch, install it first with
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then `pip install -r requirements.txt`.
- **GPU:** CUDA is used automatically when `torch.cuda.is_available()`; otherwise CPU.
- **Model download:** the checkpoint (~15 MB) downloads from Hugging Face the first time you click
  **Analyze with AI** and is cached in `~/.cache/huggingface`. After that, `HF_HUB_OFFLINE=1` works offline.
  If the download fails, the dashboard shows the error; it never falls back to fake results.

## Add sample imagery

No imagery is included. Download a few tiles from [xBD](https://xview2.org/dataset) (registration required) and
place each case in its own folder; details in [`data/samples/README.md`](data/samples/README.md):

```
data/samples/hurricane-michael_00000123/
├── hurricane-michael_00000123_pre_disaster.png    # optional, display only
├── hurricane-michael_00000123_post_disaster.png   # required, model input
├── hurricane-michael_00000123_post_disaster.json  # optional xBD labels (evaluation only)
└── case.json                                      # optional {"name", "disaster_type", "description"}
```

The disaster name, type and capture date are read from the xBD label metadata when available. You can also
upload your own pre/post images in the dashboard without adding any files.

## Using the dashboard

The app has three views, switched from the navigation bar at the top.

1. **New Assessment**: enter the disaster name and type, choose a saved case or upload an after-disaster image
   (required) and a before-disaster image (optional), check the previews, then click **Analyze with AI**.
2. **Response Overview** opens when the assessment finishes:
   - four key indicators (high-priority zones, damaged building area, severe damage, most affected zone);
   - the **priority map**, the image divided into nine zones tinted by priority. Hover a zone for its figures
     and click it to select it;
   - a ranked list of zones (name, priority, severity) beside the map, plus details for the selected zone;
   - a plain-language assessment summary and exports: damage overlay (PNG), zone ranking (CSV) and
     assessment summary (TXT).
3. **Detailed Analysis**: large before/after imagery with the AI damage overlay and an opacity control, damage
   breakdown across the image and by zone, and before/after close-ups with statistics for any zone.

**Advanced settings** (top right) holds the priority levels and information about the AI model. On the
Detailed Analysis view, **Technical details** shows processing information, the scoring method and, for
cases with xBD labels, an accuracy check against the reference annotations (IoU, precision, recall, confusion
matrix). The annotations are used only for that check and are never shown as results.

Priority colours: red = high, amber = medium, teal = low, grey = no buildings detected. Damage colours: green
= no visible damage, yellow = minor, orange = major, red = destroyed. Every colour also has a text label.

## Scoring

```
score = (1 × Minor px + 3 × Major px + 5 × Destroyed px) ÷ predicted building px
predicted building px = No Damage + Minor + Major + Destroyed      (background excluded)
```

| Score | Priority |
|---|---|
| ≥ 2.5 | High |
| 1.0 – < 2.5 | Medium |
| < 1.0 | Low |
| no building pixels | Insufficient Data |

Thresholds are preliminary design choices, not validated standards. Change them under **Advanced settings** in
the dashboard or edit `DEFAULT_THRESHOLDS` / `DAMAGE_WEIGHTS` in `ranking.py`. Zones are image sectors A–I (row by row), not
geographic areas. All percentages are pixel proportions, **not building counts**.

## Command-line tools

```bash
# Run the real model on one image; saves mask, overlay and priority map to outputs/ and prints the ranking
python verify_inference.py data/samples/<case>/<case>_post_disaster.png

# Evaluate the model against xBD labels for all local cases that have them
python evaluate.py
```

## Tests

```bash
pytest                                             # offline: ranking, image processing, data loading,
                                                   # evaluation, inference pipeline, dashboard smoke test
RUN_MODEL_TESTS=1 pytest -s                        # also downloads and runs the real checkpoint,
                                                   # including an end-to-end dashboard test on a sample case
```

Ranking tests use small hand-made masks. `tests/test_model.py` uses a tiny randomly initialised SegFormer only
to check shapes, resizing and caching offline. Neither is ever used as a prediction in the app.

## Project structure

```
app.py               Streamlit app: New Assessment, Response Overview, Detailed Analysis
model.py             Cached checkpoint loading (CUDA/CPU) and inference
image_processing.py  Preprocessing, damage overlay, priority-map rendering
ranking.py           3×3 zones, scores, priorities, explanations, summary
data_loader.py       Local case discovery, xBD label/metadata parsing
evaluation.py        Ground-truth rasterisation and metrics (separate from inference)
verify_inference.py  CLI: real inference check on one image
evaluate.py          CLI: evaluation on local labelled cases
tests/               pytest suite
data/samples/        Local disaster cases (add your own xBD tiles)
.streamlit/          Light theme (colour palette)
```

## Model

| | |
|---|---|
| Repository | [`BiliSakura/xBD-post-bda-models-transformers`](https://huggingface.co/BiliSakura/xBD-post-bda-models-transformers) |
| Checkpoint | `segformer-b0/xbd-baseline/best` (SegFormer, MiT-B0) |
| Reported validation micro IoU | 40.6 % (B0); larger variants up to 47.7 % |
| Classes | 0 background, 1 no damage, 2 minor, 3 major, 4 destroyed |
| Preprocessing | RGB → [0, 1] → ImageNet mean/std (no `preprocessor_config.json` ships with the checkpoint) |
| Input size | trained on 1024 × 1024 xBD tiles; larger images are downscaled to a 1024 px long side for the forward pass, and the mask is always returned at the original size |
| License | CC BY-NC-SA 4.0 (non-commercial), same as xBD |

Expect noisy predictions, especially between Minor and Major damage, and weaker results on imagery that differs
from xBD in resolution or sensor.
