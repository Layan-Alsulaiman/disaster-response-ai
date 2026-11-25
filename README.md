<p align="center">
  <img src="assets/logo.svg" width="96" alt="Disaster Damage Intelligence logo" />
</p>

<h1 align="center">Disaster Damage Intelligence</h1>
<p align="center"><strong>AI-powered building-damage assessment and response prioritization from satellite imagery</strong></p>

<p align="center">
  Hackathon MVP · SegFormer-B0 (xBD) · Streamlit
</p>

---

## Overview

After an earthquake, flood, hurricane or wildfire, emergency teams need to know **where** to look first, often
before any ground survey is possible. **Disaster Damage Intelligence** turns a single post-disaster satellite
image into a ranked map of response zones: it segments visible building damage with a pretrained computer-vision
model, splits the image into a 3×3 grid, scores each zone from the predicted damage, and ranks zones as **High /
Medium / Low** priority — with a plain-language explanation for every ranking.

It is a decision-support demonstration, not an operational system: it ranks *visual damage severity* only, and
does not estimate casualties, population exposure, or real-world rescue urgency.

## Problem Statement

Satellite imagery after a disaster is often the fastest information available, but reviewing it manually across
a large area is slow, and treats every part of the image as equally urgent. Response teams need a quick, visual
way to triage affected areas — to see at a glance which sections of a scene show the most severe damage, so
limited time and resources go where they matter most first.

## Proposed Solution

The application runs a pretrained SegFormer-B0 model (fine-tuned on the xBD dataset) on the post-disaster image,
producing a pixel-level damage map across five classes: background, no damage, minor, major and destroyed. It
then divides that map into a 3×3 grid of zones, computes a transparent severity score for each zone from its
predicted damage mix, and classifies zones into priority tiers. The result is presented on an interactive
dashboard built for a non-technical responder: a priority map, ranked zone list, plain-language findings, and
exportable results — with the underlying model and data details available on request, not forced onto the main
view.

## Key Features

- **Real AI inference** — a pretrained SegFormer-B0 checkpoint, fine-tuned on xBD, run live on the input image (no mocked or hardcoded predictions).
- **5-class damage segmentation** — background, no damage, minor, major, destroyed.
- **Automatic zone division** — the image is split into a 3×3 grid of response zones.
- **Transparent priority ranking** — each zone gets a severity score and a High / Medium / Low / Insufficient Data classification, with the formula and thresholds visible in Advanced Settings.
- **Plain-language explanations** — every zone's ranking comes with a short, human-readable reason, not just a number.
- **Before/after comparison** — the pre-disaster image (when available) is shown alongside the post-disaster analysis for visual context; it is not used as a model input.
- **Interactive priority map** — click any zone to inspect it, switch between priority, damage-detection and satellite-only layers.
- **Exportable results** — damage overlay (PNG), zone ranking (CSV) and assessment summary (TXT).
- **Reference-only accuracy check** — for sample cases that ship with xBD ground-truth labels, a separate evaluation panel compares predictions against them; this is never used as, or mixed into, the live results.
- **Works with local samples or uploads** — use a bundled xBD case or upload your own pre/post images.

## System Architecture & Workflow

```
 Post-disaster image ──► Preprocessing ──► SegFormer-B0 (xBD) ──► 5-class damage mask
                                                                        │
 Pre-disaster image ──────────────► shown for visual comparison only   │
                                                                        ▼
                                                              3×3 zone division
                                                                        │
                                                                        ▼
                                                          Zone severity scoring
                                                                        │
                                                                        ▼
                                                     Priority classification (High/Medium/Low)
                                                                        │
                                                                        ▼
                                                           Interactive dashboard
```

The model takes a **single** post-disaster image as input and performs semantic segmentation — it does **not**
perform two-image change detection. The pre-disaster image, when provided, is displayed only so a user can
visually compare before and after; it never reaches the model.

## Technology Stack

| Layer | Technology |
|---|---|
| AI model | [SegFormer-B0](https://huggingface.co/docs/transformers/model_doc/segformer) via Hugging Face Transformers, PyTorch |
| Image processing | Pillow, NumPy, OpenCV |
| Zone ranking / analysis | Pandas, NumPy |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Language | Python 3.11+ |

## Dashboard Overview

A dark, command-center style interface with three views, reached from the navigation bar under the header:

1. **New Assessment** — pick a saved case or upload a post-disaster image (required) and a pre-disaster image
   (optional), enter the disaster name and type, then run the analysis.
2. **Response Overview** — the main results view: summary cards (high-priority zone count, damaged building
   area, severe damage, most affected zone), an interactive priority map with switchable layers, a ranked zone
   list, a panel for the selected zone, plain-language key findings, and export buttons.
3. **Detailed Analysis** — tabbed secondary views: before/after imagery with overlay control, a per-zone
   close-up, an image-wide damage breakdown, and technical details (model info, scoring method, and the
   reference-only accuracy check for labelled samples).

Priority colors: coral = high, amber = medium, teal = low, slate = no buildings detected. Damage colors: green =
no visible damage, yellow = minor, orange = major, red = destroyed.

### Screenshots

*Screenshots of the running application go here.*

<!--
![New Assessment](docs/screenshots/new-assessment.png)
![Response Overview](docs/screenshots/response-overview.png)
![Detailed Analysis](docs/screenshots/detailed-analysis.png)
-->

## Installation & Execution

Requires **Python 3.11+** (3.10–3.13 should also work).

### Windows

```powershell
git clone https://github.com/Layan-Alsulaiman/disaster-response-ai.git
cd disaster-response-ai

python -m venv .venv
.venv\Scripts\activate

# CPU-only (recommended unless you have a CUDA GPU set up):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

streamlit run app.py
```

### macOS / Linux

```bash
git clone https://github.com/Layan-Alsulaiman/disaster-response-ai.git
cd disaster-response-ai

python -m venv .venv
source .venv/bin/activate

# CPU-only (recommended unless you have a CUDA GPU set up):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

streamlit run app.py
```

On either platform, CUDA is used automatically when available (`torch.cuda.is_available()`); otherwise the
model runs on CPU. The first time you click **Analyze with AI**, the checkpoint (~15 MB) downloads from Hugging
Face and is cached locally (`~/.cache/huggingface`); after that, it works offline with `HF_HUB_OFFLINE=1`. If the
download fails, the dashboard reports the error — it never falls back to fake results.

No sample imagery ships with the repository. Download a few tiles from [xBD](https://xview2.org/dataset)
(registration required) and add them under `data/samples/` — see
[`data/samples/README.md`](data/samples/README.md) for the expected layout — or simply upload your own
pre/post images directly in the dashboard.

## Dataset & Pretrained Model

| | |
|---|---|
| Dataset | [xBD (xView2)](https://xview2.org/dataset) — pre/post-disaster satellite imagery with building damage annotations |
| Model repository | [`BiliSakura/xBD-post-bda-models-transformers`](https://huggingface.co/BiliSakura/xBD-post-bda-models-transformers) |
| Checkpoint | `segformer-b0/xbd-baseline/best` (SegFormer, MiT-B0 backbone) |
| Classes | 0 background, 1 no damage, 2 minor, 3 major, 4 destroyed |
| Preprocessing | RGB → [0, 1] → ImageNet mean/std normalization |
| Input size | trained on 1024×1024 xBD tiles; larger images are downscaled to a 1024px long side for inference, and the output mask is always upsampled back to the original image size |
| License | CC BY-NC-SA 4.0 (non-commercial), same as xBD |

This model is **used as published**: it is not trained or fine-tuned further in this project. Reported
validation quality for this checkpoint is modest (micro IoU around 40%), so predictions — especially the
boundary between Minor and Major damage — should be read as indicative, not authoritative, particularly on
imagery that differs from xBD in resolution or sensor.

## Damage Assessment & Zone Prioritization Methodology

The post-disaster image is divided into a 3×3 grid of nine zones (A–I, row by row). For each zone, the
predicted pixels are weighted by severity and combined into a single score:

```
score = (1 × Minor px + 3 × Major px + 5 × Destroyed px) ÷ predicted building px
predicted building px = No Damage + Minor + Major + Destroyed px   (background excluded)
```

| Score | Priority |
|---|---|
| ≥ 2.5 | High |
| 1.0 – < 2.5 | Medium |
| < 1.0 | Low |
| no building pixels predicted | Insufficient Data |

These thresholds and weights are preliminary design choices made for this MVP, not validated emergency-response
standards; they are adjustable in **Advanced Settings** in the dashboard, or directly in `ranking.py`. Zones are
image sectors, not surveyed geographic areas, and every percentage shown refers to **predicted building pixel
area**, not a count of buildings.

## Project Structure

```
app.py               Streamlit dashboard: New Assessment, Response Overview, Detailed Analysis
model.py             Cached SegFormer-B0 checkpoint loading (CUDA/CPU) and inference
image_processing.py  Preprocessing, damage overlay, priority-map rendering
ranking.py           Zone division, scoring, priority classification, explanations
data_loader.py       Local case discovery, xBD label/metadata parsing
evaluation.py        Ground-truth rasterization and accuracy metrics (kept separate from inference)
verify_inference.py  CLI: run real inference on one image and inspect the result
evaluate.py          CLI: evaluate the model against local xBD-labelled cases
tests/               pytest suite
data/samples/        Local disaster cases (add your own xBD tiles here)
.streamlit/          Dashboard theme configuration
assets/logo.svg      Application logo (original artwork)
doc/project-spec     Full project specification
```

## Tests

```bash
pytest                          # offline: ranking, image processing, data loading, evaluation, inference pipeline
RUN_MODEL_TESTS=1 pytest -s     # also downloads and runs the real checkpoint end to end
```

---

<p align="center"><sub>Decision-support demonstration for a hackathon submission — not an operational emergency-response system.</sub></p>
