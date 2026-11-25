"""Streamlit dashboard: AI disaster damage assessment & zone priority ranking.

Run with:  streamlit run app.py

Every damage statistic and priority shown here is computed from a real forward
pass of the pretrained SegFormer-B0 xBD checkpoint on the post-disaster image.
The pre-disaster image is displayed for visual comparison only.
"""

from __future__ import annotations

import hashlib
import html
import io
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import data_loader as dl
import ranking as rk
from image_processing import CLASS_COLORS, CLASS_NAMES, crop_box, draw_zone_grid, load_rgb_image, overlay_mask

MODEL_REPO = "BiliSakura/xBD-post-bda-models-transformers"
MODEL_CHECKPOINT = "segformer-b0/xbd-baseline/best"

st.set_page_config(page_title="Disaster Damage Intelligence", page_icon="🛰️", layout="wide")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; max-width: 1500px;}
      .hero {background: linear-gradient(120deg, #1a2540 0%, #2a1320 100%);
             border: 1px solid #27324a; border-radius: 16px; padding: 22px 28px; margin-bottom: 18px;}
      .hero h1 {font-size: 1.9rem; margin: 0 0 4px 0; color: #f8fafc;}
      .hero p {color: #9fb0c8; margin: 0;}
      .section {font-size: 1.15rem; font-weight: 700; color: #f1f5f9; letter-spacing: .02em;
                margin: 26px 0 10px 0; padding-left: 10px; border-left: 4px solid #ef4444;}
      .section span {color: #7d8ba3; font-weight: 500; font-size: .85rem; margin-left: 8px;}
      .card {background: #131c2e; border: 1px solid #24304a; border-radius: 14px; padding: 16px 18px;}
      .kpi {background: #131c2e; border: 1px solid #24304a; border-radius: 14px; padding: 14px 16px;
            height: 100%;}
      .kpi .label {color: #8b9ab3; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em;}
      .kpi .value {color: #f8fafc; font-size: 1.75rem; font-weight: 700; line-height: 1.25;}
      .kpi .sub {color: #7d8ba3; font-size: .78rem;}
      .badge {display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .78rem;
              font-weight: 600; margin-right: 6px; border: 1px solid #334155; color: #cbd5e1;}
      .chip {display: inline-flex; align-items: center; margin: 0 14px 6px 0; color: #cbd5e1; font-size: .85rem;}
      .chip i {width: 14px; height: 14px; border-radius: 3px; display: inline-block; margin-right: 6px;
               border: 1px solid rgba(255,255,255,.25);}
      .placeholder {background: repeating-linear-gradient(45deg, #111827, #111827 10px, #131c2e 10px, #131c2e 20px);
                    border: 1px dashed #334155; border-radius: 12px; min-height: 260px; display: flex;
                    align-items: center; justify-content: center; text-align: center; color: #7d8ba3; padding: 20px;}
      .note {color: #8b9ab3; font-size: .82rem;}
      .stButton button {width: 100%;}
    </style>
    """,
    unsafe_allow_html=True,
)


def section(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="section">{title}<span>{subtitle}</span></div>', unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "") -> str:
    return f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'


def hex_color(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


PRIORITY_HEX = {p: hex_color(c) for p, c in rk.PRIORITY_COLORS.items()}
CLASS_HEX = {c: hex_color(CLASS_COLORS[c]) for c in range(1, 5)}
PLOT_LAYOUT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                   margin=dict(l=10, r=10, t=36, b=10), font=dict(color="#cbd5e1"))

# ---------------------------------------------------------------------------
# Model (loaded once per process) and inference (cached per image)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_model():
    """Import and load the pretrained checkpoint once. Exceptions are not cached."""
    import model as seg  # imported lazily so the dashboard opens even if torch is missing

    device = seg.get_device()
    seg.load_model(device)
    return seg, device


@st.cache_data(show_spinner=False, max_entries=16)
def run_inference(post_bytes: bytes) -> dict:
    seg, device = get_model()
    image = load_rgb_image(io.BytesIO(post_bytes))
    pred = seg.predict(image, device=device)
    return {
        "mask": pred.mask,
        "device": pred.device,
        "seconds": pred.inference_seconds,
        "input_size": pred.model_input_size,
        "size": pred.original_size,
    }


def image_key(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


# ---------------------------------------------------------------------------
# Sidebar: configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("**Priority thresholds** (damage score 0–5)")
    high = st.slider("High priority if score ≥", 0.1, 5.0, rk.DEFAULT_THRESHOLDS["high"], 0.1)
    medium = st.slider("Medium priority if score ≥", 0.0, 5.0, rk.DEFAULT_THRESHOLDS["medium"], 0.1)
    try:
        thresholds = rk.validate_thresholds({"high": high, "medium": medium})
    except ValueError:
        st.error("Medium threshold must be lower than the High threshold. Using defaults.")
        thresholds = dict(rk.DEFAULT_THRESHOLDS)
    st.caption("Preliminary design choices, not validated emergency-response standards.")

    with st.expander("Scoring formula", expanded=False):
        st.markdown(
            "Score = (1 × Minor px + 3 × Major px + 5 × Destroyed px) ÷ predicted building px\n\n"
            "Building px = No Damage + Minor + Major + Destroyed (background excluded). "
            "Zones with no predicted building pixels are **Insufficient Data**."
        )
        st.dataframe(
            pd.DataFrame({"Class": [CLASS_NAMES[c] for c in range(1, 5)],
                          "Weight": [rk.DAMAGE_WEIGHTS[c] for c in range(1, 5)]}),
            hide_index=True,
        )

    st.markdown("---")
    st.markdown("**AI model**")
    st.caption(f"SegFormer-B0 · `{MODEL_REPO}`\n\n`{MODEL_CHECKPOINT}`\n\n"
               "Input: post-disaster RGB image. License: CC BY-NC-SA 4.0.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
      <h1>🛰️ Disaster Damage Intelligence</h1>
      <p>AI building-damage segmentation on post-disaster satellite imagery, turned into ranked response zones.
      Decision-support demonstration — not an operational emergency-response system.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Section 1: disaster selection
# ---------------------------------------------------------------------------

section("1 · Disaster Selection", "choose a local case or upload your own imagery")

cases = dl.discover_cases()
sources = ["Sample cases", "Upload images"] if cases else ["Upload images"]
left, right = st.columns([1.1, 1], gap="large")

case_name, case_type, case_date, location = "", "", "", None
pre_bytes = post_bytes = None
labels_path = None

with left:
    source = st.radio("Image source", sources, horizontal=True)
    if source == "Sample cases":
        labels = [f"{c.name} · {c.disaster_type}" for c in cases]
        idx = st.selectbox("Disaster case", range(len(cases)), format_func=lambda i: labels[i])
        case = cases[idx]
        case_name, case_type, case_date = case.name, case.disaster_type, case.capture_date
        post_bytes = case.post_path.read_bytes()
        pre_bytes = case.pre_path.read_bytes() if case.pre_path else None
        labels_path = case.labels_path if case.has_labels else None
        if labels_path:
            try:
                location = dl.tile_location(dl.load_json(labels_path))
            except (OSError, ValueError):
                location = None
        if case.description:
            st.caption(case.description)
    else:
        if not cases:
            st.info("No local sample cases found in `data/samples/`. Upload images below, or add xBD cases "
                    "as described in `data/samples/README.md`.")
        case_name = st.text_input("Disaster name", "Uploaded case")
        case_type = st.selectbox("Disaster type", dl.UPLOAD_DISASTER_TYPES)
        u1, u2 = st.columns(2)
        pre_file = u1.file_uploader("Pre-disaster image (optional)", type=["png", "jpg", "jpeg", "tif", "tiff"])
        post_file = u2.file_uploader("Post-disaster image (model input)", type=["png", "jpg", "jpeg", "tif", "tiff"])
        pre_bytes = pre_file.getvalue() if pre_file else None
        post_bytes = post_file.getvalue() if post_file else None

post_img = pre_img = None
for raw, which in ((post_bytes, "post"), (pre_bytes, "pre")):
    if raw is None:
        continue
    try:
        img = load_rgb_image(io.BytesIO(raw))
    except Exception as exc:  # unreadable upload
        st.error(f"Could not read the {which}-disaster image: {exc}")
        img = None
    if which == "post":
        post_img = img
    else:
        pre_img = img

current_key = image_key(post_bytes) if post_img is not None else None
analysis = st.session_state.get("analysis")
if analysis and analysis["key"] != current_key:
    analysis = None

with right:
    badges = f'<span class="badge">{html.escape(case_type or "Type not set")}</span>'
    if case_date:
        badges += f'<span class="badge">Captured {case_date}</span>'
    if post_img is not None:
        badges += f'<span class="badge">{post_img.size[0]}×{post_img.size[1]} px</span>'
    loc = (f"<div class='note' style='margin-top:8px'>Tile location from xBD annotation metadata: "
           f"{location[0]:.4f}°, {location[1]:.4f}°</div>") if location else ""
    status = "✅ AI analysis complete" if analysis else ("Ready to analyze" if post_img is not None else
                                                         "Waiting for a post-disaster image")
    st.markdown(
        f'<div class="card"><div class="note">Selected case</div>'
        f'<div style="font-size:1.35rem;font-weight:700;color:#f8fafc;margin:2px 0 8px">{html.escape(case_name or "—")}</div>'
        f'{badges}{loc}<div class="note" style="margin-top:10px">Status: {status}</div></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    analyze = st.button("🔍 Analyze with AI", type="primary", disabled=post_img is None)

if analyze and post_img is not None:
    try:
        with st.spinner("Loading SegFormer-B0 (first run downloads ~15 MB) and running inference…"):
            out = run_inference(post_bytes)
        st.session_state["analysis"] = {"key": current_key, **out}
        st.session_state.pop("zone", None)
    except ImportError as exc:
        st.error(f"PyTorch / Transformers are not installed, so the AI model cannot run: {exc}\n\n"
                 "Install the dependencies with `pip install -r requirements.txt` and restart the app.")
    except Exception as exc:  # model download/load failures, out-of-memory, bad images…
        st.error(f"AI analysis failed: {exc}")
    else:
        st.rerun()

# ---------------------------------------------------------------------------
# Section 2: satellite image analysis
# ---------------------------------------------------------------------------

section("2 · Satellite Image Analysis", "the model analyzes the post-disaster image only")

alpha = st.slider("Damage overlay opacity", 0.0, 1.0, 0.55, 0.05)
c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    st.markdown("**Before disaster**")
    if pre_img is not None:
        st.image(pre_img)
    else:
        st.markdown('<div class="placeholder">No pre-disaster image provided</div>', unsafe_allow_html=True)
with c2:
    st.markdown("**After disaster** · model input")
    if post_img is not None:
        st.image(post_img)
    else:
        st.markdown('<div class="placeholder">Select a case or upload a post-disaster image</div>',
                    unsafe_allow_html=True)
with c3:
    st.markdown("**AI damage overlay**")
    if analysis is not None:
        overlay = overlay_mask(post_img, analysis["mask"], alpha=alpha)  # reused below
        st.image(overlay)
    else:
        st.markdown('<div class="placeholder">Click “Analyze with AI” to run the SegFormer model</div>',
                    unsafe_allow_html=True)

legend = "".join(f'<span class="chip"><i style="background:{CLASS_HEX[c]}"></i>{CLASS_NAMES[c]}</span>'
                 for c in range(1, 5))
st.markdown(legend + '<span class="chip"><i style="background:transparent"></i>Background (transparent)</span>',
            unsafe_allow_html=True)
st.markdown('<div class="note">The pre-disaster image is shown for visual comparison only. The model segments '
            'visible building damage in the post-disaster image; it does not perform before/after change '
            'detection.</div>', unsafe_allow_html=True)

if analysis is None:
    st.stop()

# ---------------------------------------------------------------------------
# Results (everything below comes from the real prediction)
# ---------------------------------------------------------------------------

mask = analysis["mask"]
result = rk.rank_zones(mask, thresholds)
stats = rk.overall_statistics(result.overall_counts)
n_with_data = sum(1 for z in result.zones if z.score is not None)

# ---------------------------------------------------------------------------
# Section 3: damage overview
# ---------------------------------------------------------------------------

section("3 · Damage Overview", "proportions of predicted building pixels — not building counts")

k = st.columns(5, gap="small")
cards = [
    ("Visibly damaged", pct(stats["damaged_share"]), "of predicted building pixels"),
    ("Minor damage", pct(stats["minor_share"]), "of predicted building pixels"),
    ("Major damage", pct(stats["major_share"]), "of predicted building pixels"),
    ("Destroyed", pct(stats["destroyed_share"]), "of predicted building pixels"),
    ("Zones analyzed", f"{len(result.zones)}", f"{n_with_data} with building pixels"),
]
for col, (label, value, sub) in zip(k, cards):
    col.markdown(kpi(label, value, sub), unsafe_allow_html=True)

g1, g2 = st.columns([1, 1.3], gap="large")
with g1:
    shares = [stats["no_damage_share"], stats["minor_share"], stats["major_share"], stats["destroyed_share"]]
    fig = go.Figure(go.Pie(
        labels=[CLASS_NAMES[c] for c in range(1, 5)], values=[result.overall_counts[c] for c in range(1, 5)],
        hole=0.6, marker=dict(colors=[CLASS_HEX[c] for c in range(1, 5)]), sort=False,
        textinfo="percent", hovertemplate="%{label}: %{value:,} px (%{percent})<extra></extra>",
    ))
    fig.update_layout(**PLOT_LAYOUT, height=300, title="Damage mix of predicted building pixels",
                      annotations=[dict(text=f"{pct(stats['building_coverage'])}<br>building px",
                                        showarrow=False, font=dict(size=14, color="#e5e7eb"))])
    if stats["building_pixels"] == 0:
        st.warning("The model predicted no building pixels in this image.")
    else:
        st.plotly_chart(fig, config={"displayModeBar": False})
with g2:
    grid_zones = result.grid_order()
    fig = go.Figure()
    for c in range(1, 5):
        fig.add_bar(name=CLASS_NAMES[c], x=[z.letter for z in grid_zones],
                    y=[100 * z.share(c) for z in grid_zones], marker_color=CLASS_HEX[c],
                    hovertemplate="Zone %{x}: %{y:.1f}%<extra>" + CLASS_NAMES[c] + "</extra>")
    fig.update_layout(**PLOT_LAYOUT, barmode="stack", height=300, title="Damage mix per zone (% of building px)",
                      yaxis=dict(range=[0, 100], ticksuffix="%"), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, config={"displayModeBar": False})

st.markdown(
    f'<div class="note">Inference: {analysis["seconds"]:.2f} s on {analysis["device"].upper()} · '
    f'model input {analysis["input_size"][0]}×{analysis["input_size"][1]} px · mask upsampled to '
    f'{analysis["size"][0]}×{analysis["size"][1]} px · buildings cover {pct(stats["building_coverage"])} '
    f'of the image (predicted).</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 4: priority map & zone selection
# ---------------------------------------------------------------------------

section("4 · Priority Map", "3 × 3 image sectors — not geographic coordinates")

ranked_names = [z.letter for z in result.zones]
if st.session_state.get("zone") not in ranked_names:
    st.session_state["zone"] = result.zones[0].letter
selected = st.session_state["zone"]

m1, m2 = st.columns([1.5, 1], gap="large")
with m1:
    st.image(draw_zone_grid(post_img, grid_zones, rk.PRIORITY_COLORS, selected=selected))
with m2:
    st.markdown("**Select a zone**")
    for r in range(rk.GRID_SIZE):
        cols = st.columns(rk.GRID_SIZE, gap="small")
        for col, z in zip(cols, grid_zones[r * rk.GRID_SIZE:(r + 1) * rk.GRID_SIZE]):
            tag = "No data" if z.score is None else f"{z.priority} {z.score:.1f}"
            col.button(f"{z.letter} · {tag}", key=f"zone_{z.letter}",
                       type="primary" if z.letter == selected else "secondary",
                       on_click=lambda letter=z.letter: st.session_state.update(zone=letter))
    st.markdown("".join(f'<span class="chip"><i style="background:{PRIORITY_HEX[p]}"></i>{p}</span>'
                        for p in rk.PRIORITY_ORDER), unsafe_allow_html=True)
    st.markdown(
        f'<div class="card" style="margin-top:8px">'
        f'<div class="kpi" style="border:none;padding:0;background:none">'
        f'<div class="label">Priority counts</div></div>'
        f'<span class="badge" style="border-color:{PRIORITY_HEX[rk.HIGH]}">High {result.count(rk.HIGH)}</span>'
        f'<span class="badge" style="border-color:{PRIORITY_HEX[rk.MEDIUM]}">Medium {result.count(rk.MEDIUM)}</span>'
        f'<span class="badge" style="border-color:{PRIORITY_HEX[rk.LOW]}">Low {result.count(rk.LOW)}</span>'
        f'<span class="badge" style="border-color:{PRIORITY_HEX[rk.INSUFFICIENT]}">'
        f'Insufficient {result.count(rk.INSUFFICIENT)}</span></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Section 5: zone ranking & details
# ---------------------------------------------------------------------------

section("5 · Zone Ranking", "sorted by damage severity score (0–5)")

df = pd.DataFrame(rk.zones_table(result))
df["Rank"] = df["Rank"].astype(str)  # ints and "–" in one column; keep a single type for Arrow


def priority_style(value):
    color = PRIORITY_HEX.get(value)
    return f"background-color: {color}33; color: {color}; font-weight: 600" if color else ""


t1, t2 = st.columns([1.35, 1], gap="large")
with t1:
    st.dataframe(
        df.style.map(priority_style, subset=["Priority"]).format(
            {"Score": lambda v: "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.2f}",
             "Building px %": "{:.1f}", "No Damage %": "{:.1f}", "Minor %": "{:.1f}",
             "Major %": "{:.1f}", "Destroyed %": "{:.1f}"}),
        hide_index=True, height=360,
    )
    scored = [z for z in result.zones if z.score is not None]
    if scored:
        fig = go.Figure(go.Bar(
            x=[z.score for z in scored], y=[z.name for z in scored], orientation="h",
            marker_color=[PRIORITY_HEX[z.priority] for z in scored],
            hovertemplate="%{y}: %{x:.2f}<extra></extra>",
        ))
        fig.add_vline(x=thresholds["high"], line_dash="dot", line_color=PRIORITY_HEX[rk.HIGH])
        fig.add_vline(x=thresholds["medium"], line_dash="dot", line_color=PRIORITY_HEX[rk.MEDIUM])
        fig.update_layout(**PLOT_LAYOUT, height=60 + 34 * len(scored), title="Damage severity score",
                          xaxis=dict(range=[0, rk.MAX_SCORE]), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, config={"displayModeBar": False})

with t2:
    z = result.by_name(selected)
    color = PRIORITY_HEX[z.priority]
    rank_txt = f"Rank #{z.rank} of {n_with_data}" if z.rank else "Not ranked"
    st.markdown(
        f'<div class="card" style="border-color:{color}">'
        f'<div class="note">Selected zone · {rank_txt}</div>'
        f'<div style="font-size:1.6rem;font-weight:800;color:#f8fafc">{z.name} '
        f'<span class="badge" style="border-color:{color};color:{color}">{z.priority}</span></div>'
        f'<div style="font-size:1.05rem;color:#cbd5e1;margin:4px 0 10px">Severity score: '
        f'<b>{"—" if z.score is None else f"{z.score:.2f} / 5"}</b></div>'
        f'<div style="color:#cbd5e1;font-size:.92rem;line-height:1.5">{z.explanation}</div></div>',
        unsafe_allow_html=True,
    )
    d1, d2 = st.columns(2)
    d1.metric("Predicted building px", f"{z.building_pixels:,}", f"{pct(z.building_coverage)} of zone",
              delta_color="off")
    d2.metric("Major + Destroyed", pct(z.severe_share), "of building px", delta_color="off")
    if z.building_pixels:
        fig = go.Figure(go.Bar(
            x=[CLASS_NAMES[c] for c in range(1, 5)], y=[z.counts[c] for c in range(1, 5)],
            marker_color=[CLASS_HEX[c] for c in range(1, 5)],
            text=[pct(z.share(c)) for c in range(1, 5)], textposition="outside",
            hovertemplate="%{x}: %{y:,} px<extra></extra>",
        ))
        fig.update_layout(**PLOT_LAYOUT, height=250, title=f"{z.name}: predicted pixels by class")
        st.plotly_chart(fig, config={"displayModeBar": False})
    st.image(crop_box(overlay, z.bbox), caption=f"{z.name} close-up (AI overlay)")

# ---------------------------------------------------------------------------
# Section 6: analysis summary
# ---------------------------------------------------------------------------

section("6 · Analysis Summary", "generated from the computed results using templates")

summary = rk.summarize(result)
st.markdown(f'<div class="card" style="font-size:1.02rem;line-height:1.6;color:#e2e8f0">{summary}</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="note" style="margin-top:10px">This ranking reflects visual damage severity in predicted '
    'building pixels only. It does not estimate casualties, population exposure or real-world rescue urgency, '
    'and pixel proportions are not building counts. Thresholds are preliminary and configurable.</div>',
    unsafe_allow_html=True,
)

dl1, dl2, dl3 = st.columns(3)
buf = io.BytesIO()
overlay.save(buf, format="PNG")
dl1.download_button("⬇️ Damage overlay (PNG)", buf.getvalue(), "damage_overlay.png", "image/png")
dl2.download_button("⬇️ Zone ranking (CSV)", df.to_csv(index=False).encode(), "zone_ranking.csv", "text/csv")
report = (f"{case_name} ({case_type})\n\n{summary}\n\n"
          + "\n".join(f"- {z.explanation}" for z in result.zones)
          + f"\n\nThresholds: High ≥ {thresholds['high']:g}, Medium ≥ {thresholds['medium']:g}. "
            f"Model: {MODEL_REPO} / {MODEL_CHECKPOINT}.\n")
dl3.download_button("⬇️ Summary (TXT)", report.encode(), "analysis_summary.txt", "text/plain")

# ---------------------------------------------------------------------------
# Optional: evaluation against xBD ground truth (reference only)
# ---------------------------------------------------------------------------

if labels_path is not None:
    with st.expander("Model evaluation against xBD ground truth (reference only)"):
        st.caption("The xBD annotation is used only to score the AI prediction. All results above come from "
                   "the model, never from the annotation.")
        if st.toggle("Compute evaluation metrics"):
            from evaluation import evaluate_prediction

            ev = evaluate_prediction(mask, dl.xbd_buildings(dl.load_json(labels_path)))

            def f(v):
                return "—" if math.isnan(v) else f"{100 * v:.1f}%"

            e1, e2, e3 = st.columns(3)
            e1.metric("Building localisation IoU", f(ev["building_iou"]))
            e2.metric("Mean damage-class IoU", f(ev["mean_damage_iou"]))
            e3.metric("Pixel accuracy", f(ev["pixel_accuracy"]))
            st.dataframe(pd.DataFrame([
                {"Class": m["class"], "IoU": f(m["iou"]), "Precision": f(m["precision"]),
                 "Recall": f(m["recall"]), "GT px": m["gt_pixels"], "Predicted px": m["pred_pixels"]}
                for m in ev["per_class"].values()
            ]), hide_index=True)
            names = [CLASS_NAMES[c] for c in range(5)]
            fig = go.Figure(go.Heatmap(z=ev["confusion_matrix"], x=names, y=names, colorscale="Reds",
                                       hovertemplate="GT %{y} → Pred %{x}: %{z:,} px<extra></extra>"))
            fig.update_layout(**PLOT_LAYOUT, height=360, title="Confusion matrix (pixels)",
                              xaxis_title="Prediction", yaxis_title="Ground truth", yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, config={"displayModeBar": False})
