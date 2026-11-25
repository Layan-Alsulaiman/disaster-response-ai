"""Disaster Response Intelligence: Streamlit front end.

Three views:
  * New Assessment       choose a case or upload imagery, run the AI assessment
  * Response Overview    priority map, ranked zones, key indicators, summary, exports
  * Detailed Analysis    large before/after imagery, damage overlay, breakdowns, zone close-ups

Run with:  streamlit run app.py

Every figure shown comes from a real forward pass of the pretrained SegFormer-B0
xBD checkpoint on the after-disaster image. The before-disaster image is shown
for visual comparison only.
"""

from __future__ import annotations

import base64
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

st.set_page_config(page_title="Disaster Response Intelligence", layout="wide",
                   initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

INK, MUTED, LINE, SURFACE, SUBTLE = "#243447", "#6B7787", "#E3E7ED", "#FFFFFF", "#EEF1F5"


def hexc(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


PRIORITY_HEX = {p: hexc(c) for p, c in rk.PRIORITY_COLORS.items()}
PRIORITY_TEXT = {rk.HIGH: "#9E3427", rk.MEDIUM: "#87591A", rk.LOW: "#2A6459", rk.INSUFFICIENT: "#5B6270"}
PRIORITY_LABEL = {rk.HIGH: "High priority", rk.MEDIUM: "Medium priority", rk.LOW: "Low priority",
                  rk.INSUFFICIENT: "No buildings detected"}
CLASS_HEX = {c: hexc(CLASS_COLORS[c]) for c in range(1, 5)}
DAMAGE_LABEL = {1: "No visible damage", 2: "Minor damage", 3: "Major damage", 4: "Destroyed"}

VIEWS = [("new", "New Assessment"), ("overview", "Response Overview"), ("detail", "Detailed Analysis")]

CHART_LAYOUT = dict(
    template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=INK, size=13), margin=dict(l=8, r=8, t=8, b=8),
    hoverlabel=dict(bgcolor=SURFACE, bordercolor=LINE, font=dict(color=INK, size=13)),
)
CHART_CONFIG = {"displayModeBar": False, "scrollZoom": False}

ss = st.session_state
ss.setdefault("view", "new")
ss.setdefault("assessment", None)
ss.setdefault("zone", None)
ss.setdefault("thr_high", rk.DEFAULT_THRESHOLDS["high"])
ss.setdefault("thr_medium", rk.DEFAULT_THRESHOLDS["medium"])
ss.setdefault("alpha", 0.55)
if ss.view != "new" and ss.assessment is None:
    ss.view = "new"

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
  #MainMenu, footer {visibility: hidden;}
  header[data-testid="stHeader"] {background: transparent;}
  .block-container {padding-top: 1.4rem; padding-bottom: 1.5rem; max-width: 1480px;}

  /* top bar */
  .brand {font-weight: 700; font-size: 1.08rem; color: #243447; line-height: 1.25;}
  .brand small {display: block; font-weight: 500; color: #6B7787; font-size: .78rem; letter-spacing: .01em;}
  [class*="st-key-nav_"] button {background: transparent; border: none; border-bottom: 2px solid transparent;
      border-radius: 0; color: #6B7787; font-weight: 600; box-shadow: none; padding: 6px 4px;}
  [class*="st-key-nav_"] button:hover {color: #243447; background: transparent; border-bottom-color: #C9D0DA;}
  [class*="st-key-nav_"] button:focus:not(:active) {color: #243447; background: transparent; box-shadow: none;}
  [class*="st-key-nav_"] button:disabled {color: #B8C0CC; background: transparent;}
  .rule {border-bottom: 1px solid #E3E7ED; margin: 0 0 20px;}

  /* typography */
  .view-title {font-size: 1.55rem; font-weight: 700; color: #243447; margin: 0; line-height: 1.25;}
  .view-sub {color: #6B7787; font-size: .92rem; margin: 4px 0 18px;}
  .label-sm {font-size: .74rem; font-weight: 700; color: #6B7787; text-transform: uppercase;
             letter-spacing: .07em; margin: 0 0 8px;}
  .muted {color: #6B7787; font-size: .84rem;}
  .tip {border-bottom: 1px dotted #9CA3AF; cursor: help;}

  /* cards: keyed containers (st-key-card_*) and HTML cards share one look */
  [class*="st-key-card_"], .card {background: #FFFFFF; border: 1px solid #E3E7ED; border-radius: 12px;
      padding: 18px 20px; box-shadow: 0 1px 2px rgba(36,52,71,.05);}
  .kpi {background: #FFFFFF; border: 1px solid #E3E7ED; border-radius: 12px; padding: 14px 18px;
        box-shadow: 0 1px 2px rgba(36,52,71,.05);}
  .kpi .k-label {font-size: .74rem; font-weight: 700; color: #6B7787; text-transform: uppercase; letter-spacing: .06em;}
  .kpi .k-value {font-size: 1.7rem; font-weight: 700; color: #243447; line-height: 1.3;}
  .kpi .k-sub {font-size: .82rem; color: #6B7787;}

  /* priority pills and chips */
  .pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600;
         vertical-align: middle;}
  .chip {display: inline-flex; align-items: center; margin: 6px 16px 0 0; font-size: .82rem; color: #243447;}
  .chip i {width: 11px; height: 11px; border-radius: 3px; margin-right: 7px; display: inline-block;}
  .tag {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600;
        background: #EEF1F5; color: #243447; margin-right: 6px;}

  /* composition bar */
  .cbar {display: flex; height: 12px; border-radius: 6px; overflow: hidden; background: #EEF1F5; margin: 6px 0 2px;}
  .cbar div {height: 100%;}
  .crow {display: flex; justify-content: space-between; font-size: .86rem; padding: 5px 0;
         border-bottom: 1px solid #F0F2F5;}
  .crow:last-child {border-bottom: none;}
  .crow span i {width: 10px; height: 10px; border-radius: 2px; display: inline-block; margin-right: 8px;}

  /* buttons */
  .stButton button, .stDownloadButton button {border-radius: 8px; font-weight: 600;}
  [class*="st-key-zbtn_"] button {justify-content: flex-start; text-align: left; background: #FFFFFF;
      border: 1px solid #E3E7ED; border-left-width: 5px; border-radius: 8px; padding: 7px 12px; min-height: 0;}
  [class*="st-key-zbtn_"] button:hover {background: #F7F8FA; border-color: #C9D0DA;}
  [class*="st-key-zbtn_"] button p {font-size: .9rem;}

  /* imagery */
  [data-testid="stImage"] img {border-radius: 10px;}
  [data-testid="stImageCaption"], [data-testid="caption"] {color: #6B7787;}

  .footer {color: #8A94A3; font-size: .78rem; border-top: 1px solid #E3E7ED; margin-top: 28px; padding-top: 10px;}
</style>
""",
    unsafe_allow_html=True,
)


def css(rules: str) -> None:
    st.markdown(f"<style>{rules}</style>", unsafe_allow_html=True)


def esc(text) -> str:
    return html.escape(str(text))


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def pill(priority: str) -> str:
    return (f'<span class="pill" style="background:{rgba(PRIORITY_HEX[priority], .16)};'
            f'color:{PRIORITY_TEXT[priority]}">{PRIORITY_LABEL[priority]}</span>')


def kpi(label: str, value: str, sub: str = "", tip: str = "") -> str:
    lab = f'<span class="tip" title="{esc(tip)}">{label}</span>' if tip else label
    return (f'<div class="kpi"><div class="k-label">{lab}</div><div class="k-value">{value}</div>'
            f'<div class="k-sub">{sub}</div></div>')


def damage_legend() -> str:
    return "".join(f'<span class="chip"><i style="background:{CLASS_HEX[c]}"></i>{DAMAGE_LABEL[c]}</span>'
                   for c in range(1, 5))


def priority_legend() -> str:
    return "".join(f'<span class="chip"><i style="background:{PRIORITY_HEX[p]}"></i>{PRIORITY_LABEL[p]}</span>'
                   for p in rk.PRIORITY_ORDER)


def composition(shares: dict[int, float], rows: bool = True) -> str:
    """Stacked bar of damage classes (shares of building area) plus a labelled breakdown."""
    bar = "".join(f'<div style="width:{100 * shares[c]:.3f}%;background:{CLASS_HEX[c]}" '
                  f'title="{DAMAGE_LABEL[c]}: {pct(shares[c])}"></div>' for c in range(1, 5))
    out = f'<div class="cbar">{bar}</div>'
    if rows:
        out += "".join(f'<div class="crow"><span><i style="background:{CLASS_HEX[c]}"></i>{DAMAGE_LABEL[c]}</span>'
                       f'<b>{pct(shares[c])}</b></div>' for c in range(1, 5))
    return out


def zone_shares(z) -> dict[int, float]:
    return {c: z.share(c) for c in range(1, 5)}


def show_error(message: str, hint: str, exc: Exception) -> None:
    st.error(f"**{message}** {hint}")
    with st.expander("Technical details"):
        st.code(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Model and inference (model loaded once per process, inference cached per image)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_model():
    import model as seg  # lazy import: the app opens even if PyTorch is missing

    device = seg.get_device()
    seg.load_model(device)
    return seg, device


@st.cache_data(show_spinner=False, max_entries=16)
def run_inference(post_bytes: bytes) -> dict:
    seg, device = get_model()
    pred = seg.predict(load_rgb_image(io.BytesIO(post_bytes)), device=device)
    return {"mask": pred.mask, "device": pred.device, "seconds": pred.inference_seconds,
            "input_size": pred.model_input_size, "size": pred.original_size}


@st.cache_data(show_spinner=False, max_entries=8)
def decode(data: bytes) -> Image.Image:
    return load_rgb_image(io.BytesIO(data))


def to_data_uri(img: Image.Image, max_side: int = 1100) -> str:
    img = img.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def enlarge(img: Image.Image, target: int = 900) -> Image.Image:
    """Upscale small crops so they fill their column (st.image never upscales)."""
    w, h = img.size
    scale = target / max(w, h)
    return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS) if scale > 1 else img


# ---------------------------------------------------------------------------
# Navigation callbacks
# ---------------------------------------------------------------------------


def go_to(view: str) -> None:
    ss.view = view


def select_zone(letter: str) -> None:
    ss.zone = letter


def inspect_zone(letter: str) -> None:
    ss.zone = letter
    ss.view = "detail"


def reset_thresholds() -> None:
    ss.thr_high = rk.DEFAULT_THRESHOLDS["high"]
    ss.thr_medium = rk.DEFAULT_THRESHOLDS["medium"]


# ---------------------------------------------------------------------------
# Top bar: brand, navigation, advanced settings
# ---------------------------------------------------------------------------

has_result = ss.assessment is not None
brand_col, nav_col, adv_col = st.columns([2.4, 5, 1.6], vertical_alignment="center")
brand_col.markdown('<div class="brand">Disaster Response Intelligence'
                   '<small>AI-assisted building damage assessment</small></div>', unsafe_allow_html=True)
with nav_col:
    for col, (key, label) in zip(st.columns(3), VIEWS):
        col.button(label, key=f"nav_{key}", on_click=go_to, args=(key,),
                   disabled=key != "new" and not has_result)
css(f".st-key-nav_{ss.view} button {{color: {INK}; border-bottom-color: {INK};}}")

with adv_col:
    with st.popover("Advanced settings"):
        st.markdown("**Priority levels**")
        st.slider("High priority from severity", 0.1, 5.0, step=0.1, key="thr_high",
                  help="Zones with a severity score at or above this value are high priority.")
        st.slider("Medium priority from severity", 0.0, 5.0, step=0.1, key="thr_medium",
                  help="Zones at or above this value (and below the high level) are medium priority.")
        st.button("Restore defaults", on_click=reset_thresholds)
        st.caption("Severity runs from 0 (no visible damage) to 5 (all destroyed). It is the average damage "
                   "level of a zone's detected building area, weighting minor damage 1, major damage 3 and "
                   "destroyed 5. The default levels (2.5 and 1.0) are preliminary, not validated standards.")
        st.markdown("**About the AI model**")
        st.caption(f"SegFormer-B0 trained on the xBD satellite damage dataset (`{MODEL_REPO}`, "
                   f"`{MODEL_CHECKPOINT}`). It analyzes the after-disaster image only and does not compare "
                   f"before and after images. Licensed CC BY-NC-SA 4.0 (non-commercial).")

try:
    thresholds = rk.validate_thresholds({"high": ss.thr_high, "medium": ss.thr_medium})
    thresholds_note = ""
except ValueError:
    thresholds = dict(rk.DEFAULT_THRESHOLDS)
    thresholds_note = "The medium level must be below the high level; default priority levels are in use."

st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
if thresholds_note:
    st.warning(thresholds_note)
if ss.pop("_toast", None):
    st.toast("Assessment complete")


# ---------------------------------------------------------------------------
# View 1: New Assessment
# ---------------------------------------------------------------------------


def view_new() -> None:
    st.markdown('<div class="view-title">New assessment</div>'
                '<div class="view-sub">Choose a disaster case or upload satellite images, then run the AI '
                'damage assessment.</div>', unsafe_allow_html=True)

    if has_result:
        a = ss.assessment
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        c1.markdown(f'<div class="muted">Current assessment: <b>{esc(a["name"])}</b>. Starting a new one '
                    f'replaces it.</div>', unsafe_allow_html=True)
        c2.button("Back to results", on_click=go_to, args=("overview",))

    cases = dl.discover_cases()
    left, right = st.columns([1, 1.3], gap="large")

    pre_bytes = post_bytes = None
    labels_path = None
    capture_date = ""
    with left:
        with st.container(key="card_setup"):
            st.markdown('<div class="label-sm">Disaster details</div>', unsafe_allow_html=True)
            options = ["Existing case", "Upload images"] if cases else ["Upload images"]
            source = st.radio("Imagery source", options, horizontal=True, label_visibility="collapsed")

            if source == "Existing case":
                idx = st.selectbox("Disaster case", range(len(cases)),
                                   format_func=lambda i: f"{cases[i].name} ({cases[i].disaster_type})")
                case = cases[idx]
                default_name, default_type = case.name, case.disaster_type
                post_bytes = case.post_path.read_bytes()
                pre_bytes = case.pre_path.read_bytes() if case.pre_path else None
                labels_path = case.labels_path if case.has_labels else None
                capture_date = case.capture_date
                key_suffix = case.case_id
            else:
                if not cases:
                    st.caption("No saved cases were found. Upload images below, or add cases to "
                               "`data/samples/` (see the README).")
                default_name, default_type, key_suffix = "", "Other", "upload"

            types = list(dl.UPLOAD_DISASTER_TYPES)
            if default_type not in types:
                default_type = "Other"
            name = st.text_input("Disaster name", default_name, key=f"name_{key_suffix}",
                                 placeholder="e.g. Hurricane Michael, Panama City")
            dtype = st.selectbox("Disaster type", types, index=types.index(default_type), key=f"type_{key_suffix}")

            if source == "Upload images":
                u1, u2 = st.columns(2)
                post_file = u1.file_uploader("After the disaster (required)",
                                             type=["png", "jpg", "jpeg", "tif", "tiff"],
                                             help="This is the image the AI analyzes.")
                pre_file = u2.file_uploader("Before the disaster (optional)",
                                            type=["png", "jpg", "jpeg", "tif", "tiff"],
                                            help="Shown for visual comparison only.")
                post_bytes = post_file.getvalue() if post_file else None
                pre_bytes = pre_file.getvalue() if pre_file else None

    post_img = pre_img = None
    for raw, which in ((post_bytes, "after"), (pre_bytes, "before")):
        if raw is None:
            continue
        try:
            img = decode(raw)
        except Exception as exc:
            show_error(f"The {which}-disaster image could not be read.", "Please upload a PNG, JPEG or TIFF image.",
                       exc)
            img = None
        if which == "after":
            post_img = img
        else:
            pre_img = img

    with right:
        with st.container(key="card_preview"):
            st.markdown('<div class="label-sm">Image preview</div>', unsafe_allow_html=True)
            if post_img is None and pre_img is None:
                st.markdown('<div class="muted">Previews appear here once images are selected.</div>',
                            unsafe_allow_html=True)
            else:
                p1, p2 = st.columns(2, gap="medium")
                with p1:
                    if pre_img is not None:
                        st.image(pre_img, caption="Before (for comparison)")
                    else:
                        st.markdown('<div class="muted">No before-disaster image</div>', unsafe_allow_html=True)
                with p2:
                    if post_img is not None:
                        st.image(post_img, caption="After (analyzed by the AI)")
                    else:
                        st.markdown('<div class="muted">An after-disaster image is required</div>',
                                    unsafe_allow_html=True)

    st.write("")
    b1, b2 = st.columns([1, 2.3], vertical_alignment="center")
    run = b1.button("Analyze with AI", type="primary", disabled=post_img is None, key="run")
    b2.markdown('<div class="muted">The AI examines the after-disaster image for visible building damage. '
                'The before-disaster image is not analyzed.</div>', unsafe_allow_html=True)

    if run and post_img is not None:
        try:
            with st.spinner("Analyzing the after-disaster image. The first run also downloads the AI model "
                            "(about 15 MB)."):
                out = run_inference(post_bytes)
        except ImportError as exc:
            show_error("The AI components are not installed on this computer.",
                       "Run `pip install -r requirements.txt`, then restart the app.", exc)
        except Exception as exc:
            show_error("The assessment could not be completed.",
                       "If this is the first run, check the internet connection so the AI model can be "
                       "downloaded, then try again.", exc)
        else:
            location = None
            if labels_path:
                try:
                    location = dl.tile_location(dl.load_json(labels_path))
                except (OSError, ValueError):
                    location = None
            ss.assessment = {
                "key": hashlib.sha1(post_bytes).hexdigest(),
                "name": name.strip() or "Untitled assessment", "type": dtype, "date": capture_date,
                "location": location, "labels_path": labels_path,
                "post_bytes": post_bytes, "pre_bytes": pre_bytes, **out,
            }
            ss.zone = None
            ss.view = "overview"
            ss["_toast"] = True
            st.rerun()


# ---------------------------------------------------------------------------
# Shared result helpers
# ---------------------------------------------------------------------------


def header(a: dict, subtitle: str) -> None:
    tags = f'<span class="tag">{esc(a["type"])}</span>'
    if a.get("date"):
        tags += f'<span class="tag">Imagery captured {esc(a["date"])}</span>'
    loc = ""
    if a.get("location"):
        lat, lon = a["location"]
        loc = (f'<span class="muted tip" title="Average position of annotated buildings in the imagery '
               f'metadata">{lat:.4f}°, {lon:.4f}°</span>')
    st.markdown(f'<div class="view-title">{esc(a["name"])}</div>'
                f'<div class="view-sub">{subtitle}&nbsp;&nbsp;{tags}{loc}</div>', unsafe_allow_html=True)


def apply_map_selection(letters: list[str]) -> None:
    """Turn a click on the priority map (read from widget state) into a zone selection."""
    try:
        points = ss["priority_map"]["selection"]["points"]
    except (KeyError, TypeError):
        points = []
    if not points:
        ss["_map_sig"] = None
        return
    sig = tuple(sorted(str(p.get("customdata", p.get("curve_number"))) for p in points))
    if sig == ss.get("_map_sig"):
        return
    ss["_map_sig"] = sig
    p = points[-1]
    letter = p.get("customdata")
    if isinstance(letter, (list, tuple)):
        letter = letter[0] if letter else None
    if letter not in letters:
        cn = p.get("curve_number")
        letter = letters[cn] if isinstance(cn, int) and 0 <= cn < len(letters) else None
    if letter:
        ss.zone = letter


def priority_map_figure(post_img: Image.Image, result: rk.RankingResult, selected: str) -> go.Figure:
    grid = result.grid_order()
    w, h = post_img.size
    base = draw_zone_grid(post_img, grid, rk.PRIORITY_COLORS, selected=selected, fill_alpha=0.48,
                          label=lambda z: "No buildings" if z.score is None else f"{z.priority} · {z.score:.1f}")

    fig = go.Figure()
    for z in grid:  # one invisible, clickable point grid per zone (trace index == grid position)
        x0, y0, x1, y1 = z.bbox
        xs, ys = np.meshgrid(np.linspace(x0, x1, 9)[1:-1], np.linspace(y0, y1, 9)[1:-1])
        if z.score is None:
            detail = "No buildings detected in this zone"
        else:
            detail = (f"{PRIORITY_LABEL[z.priority]} · severity {z.score:.1f} / 5<br>"
                      f"{pct(z.severe_share)} of building area with major damage or destroyed")
        fig.add_trace(go.Scatter(
            x=xs.ravel(), y=ys.ravel(), mode="markers", customdata=[z.letter] * xs.size,
            marker=dict(size=46, color=INK, opacity=0), name=z.name, showlegend=False,
            selected=dict(marker=dict(opacity=0)), unselected=dict(marker=dict(opacity=0)),
            hovertemplate=f"<b>{z.name}</b><br>{detail}<extra></extra>",
        ))
    fig.add_layout_image(dict(source=to_data_uri(base), xref="x", yref="y", x=0, y=0, sizex=w, sizey=h,
                              sizing="stretch", layer="below"))
    fig.update_xaxes(visible=False, range=[0, w], fixedrange=True, constrain="domain")
    fig.update_yaxes(visible=False, range=[h, 0], fixedrange=True, scaleanchor="x", scaleratio=1,
                     constrain="domain")
    fig.update_layout(**{**CHART_LAYOUT, "margin": dict(l=0, r=0, t=0, b=0)}, height=640, dragmode=False,
                      hovermode="closest", clickmode="event+select")
    return fig


# ---------------------------------------------------------------------------
# View 2: Response Overview
# ---------------------------------------------------------------------------


def view_overview(a: dict, result: rk.RankingResult) -> None:
    post_img = decode(a["post_bytes"])
    stats = rk.overall_statistics(result.overall_counts)
    scored = [z for z in result.zones if z.score is not None]
    unscored = [z for z in result.zones if z.score is None]
    grid_letters = [z.letter for z in result.grid_order()]

    apply_map_selection(grid_letters)
    if ss.zone not in grid_letters:
        ss.zone = result.zones[0].letter
    zone = result.by_name(ss.zone)

    rules = [f".st-key-zbtn_{z.letter} button {{border-left-color: {PRIORITY_HEX[z.priority]};}}" for z in scored]
    rules.append(f".st-key-zbtn_{zone.letter} button {{background: {SUBTLE}; border-color: {INK};"
                 f"border-left-color: {PRIORITY_HEX[zone.priority]};}}")
    css("\n".join(rules))

    header(a, "Response overview")

    top = scored[0] if scored else None
    k = st.columns(4, gap="medium")
    k[0].markdown(kpi("High-priority zones", f"{result.count(rk.HIGH)}", f"of {len(result.zones)} zones",
                      "Zones whose severity reached the high-priority level."), unsafe_allow_html=True)
    k[1].markdown(kpi("Damaged building area", pct(stats["damaged_share"]) if scored else "—",
                      "shows visible damage",
                      "Share of the building area detected by the AI that shows minor damage, major damage "
                      "or destruction. This is an area share, not a count of buildings."), unsafe_allow_html=True)
    k[2].markdown(kpi("Severe damage", pct(stats["major_share"] + stats["destroyed_share"]) if scored else "—",
                      "major damage or destroyed",
                      "Share of the detected building area classified as major damage or destroyed."),
                  unsafe_allow_html=True)
    k[3].markdown(kpi("Most affected zone", top.name if top else "—",
                      f"{PRIORITY_LABEL[top.priority]} · severity {top.score:.1f} / 5" if top
                      else "no buildings detected",
                      "The zone with the highest severity score."), unsafe_allow_html=True)
    st.write("")

    map_col, side_col = st.columns([1.7, 1], gap="large")
    with map_col:
        with st.container(key="card_map"):
            st.markdown('<div class="label-sm">Priority map</div>', unsafe_allow_html=True)
            st.plotly_chart(priority_map_figure(post_img, result, zone.letter), config=CHART_CONFIG,
                            on_select="rerun", selection_mode="points", key="priority_map")
            st.markdown(priority_legend(), unsafe_allow_html=True)
            st.markdown('<div class="muted" style="margin-top:6px">Click a zone to inspect it. Zones are equal '
                        'sections of the image, not surveyed geographic areas.</div>', unsafe_allow_html=True)

    with side_col:
        with st.container(key="card_rank"):
            st.markdown('<div class="label-sm">Zones by severity</div>', unsafe_allow_html=True)
            if not scored:
                st.markdown('<div class="muted">No buildings were detected, so no zone could be ranked.</div>',
                            unsafe_allow_html=True)
            for z in scored:
                st.button(f"**{z.rank}. {z.name}**  ·  {z.priority}  ·  {z.score:.1f} / 5", key=f"zbtn_{z.letter}",
                          on_click=select_zone, args=(z.letter,))
            if unscored:
                st.markdown(f'<div class="muted" style="margin-top:4px">No buildings detected: '
                            f'{", ".join(z.name for z in unscored)}</div>', unsafe_allow_html=True)

        st.write("")
        with st.container(key="card_zone"):
            st.markdown('<div class="label-sm">Selected zone</div>', unsafe_allow_html=True)
            rank = f"Rank {zone.rank} of {len(scored)}" if zone.rank else "Not ranked"
            sev = (f'<span class="tip" title="0 = no visible damage, 5 = all detected building area destroyed">'
                   f'Severity <b>{zone.score:.1f} / 5</b></span> · {rank}') if zone.score is not None else rank
            st.markdown(
                f'<div style="font-size:1.3rem;font-weight:700;color:{INK}">{zone.name}&nbsp; {pill(zone.priority)}'
                f'</div><div class="muted" style="margin:4px 0 10px">{sev}</div>'
                f'<div style="font-size:.93rem;line-height:1.5;margin-bottom:8px">{esc(rk.zone_insight(zone))}</div>'
                + (composition(zone_shares(zone), rows=False) + damage_legend() if zone.score is not None else ""),
                unsafe_allow_html=True,
            )
            st.button("Inspect zone in detail", key="inspect", on_click=inspect_zone, args=(zone.letter,))

    st.write("")
    with st.container(key="card_summary"):
        s1, s2 = st.columns([2.2, 1], gap="large")
        with s1:
            st.markdown('<div class="label-sm">Assessment summary</div>'
                        f'<div style="font-size:1rem;line-height:1.6">{esc(rk.plain_summary(result))}</div>'
                        '<div class="muted" style="margin-top:10px">AI-assisted estimate from satellite imagery for '
                        'decision support. Not a verified field assessment or an operational emergency '
                        'recommendation; it does not estimate casualties or population exposure.</div>',
                        unsafe_allow_html=True)
        with s2:
            st.markdown('<div class="label-sm">Export</div>', unsafe_allow_html=True)
            export_buttons(a, result, post_img)


def export_buttons(a: dict, result: rk.RankingResult, post_img: Image.Image) -> None:
    buf = io.BytesIO()
    overlay_mask(post_img, a["mask"], alpha=ss.alpha).save(buf, format="PNG")
    st.download_button("Damage overlay (PNG)", buf.getvalue(), "damage_overlay.png", "image/png")

    df = pd.DataFrame(rk.zones_table(result)).rename(columns={
        "Score": "Severity (0-5)", "Building px %": "Building area (% of zone)",
        "No Damage %": "No visible damage (% of building area)", "Minor %": "Minor damage (% of building area)",
        "Major %": "Major damage (% of building area)", "Destroyed %": "Destroyed (% of building area)",
    })
    df["Priority"] = df["Priority"].replace({rk.INSUFFICIENT: PRIORITY_LABEL[rk.INSUFFICIENT]})
    st.download_button("Zone ranking (CSV)", df.to_csv(index=False).encode(), "zone_ranking.csv", "text/csv")

    lines = [a["name"], f"Type: {a['type']}" + (f" · Imagery captured {a['date']}" if a.get("date") else ""), "",
             rk.plain_summary(result), "", "Zones by severity:"]
    for z in result.zones:
        score = f"severity {z.score:.1f} / 5" if z.score is not None else "not scored"
        lines.append(f"- {z.name}: {PRIORITY_LABEL[z.priority]}, {score}. {rk.zone_insight(z)}")
    lines += ["", f"Priority levels: high from severity {result.thresholds['high']:g}, medium from "
                  f"{result.thresholds['medium']:g}. Severity weights: minor 1, major 3, destroyed 5.",
              "Percentages are shares of the building area detected by the AI, not counts of buildings.",
              "AI-assisted estimate for decision support; not a verified field assessment."]
    st.download_button("Assessment summary (TXT)", "\n".join(lines).encode(), "assessment_summary.txt",
                       "text/plain")


# ---------------------------------------------------------------------------
# View 3: Detailed Analysis
# ---------------------------------------------------------------------------


def zone_chart(result: rk.RankingResult) -> go.Figure:
    grid = result.grid_order()
    ticks = [f"{z.name}<br><span style='color:{PRIORITY_TEXT[z.priority]}'>"
             f"{'No data' if z.score is None else z.priority}</span>" for z in grid]
    fig = go.Figure()
    for c in range(1, 5):
        fig.add_bar(name=DAMAGE_LABEL[c], x=ticks, y=[100 * z.share(c) for z in grid], marker_color=CLASS_HEX[c],
                    hovertemplate=f"%{{x}}<br>{DAMAGE_LABEL[c]}: %{{y:.0f}}% of building area<extra></extra>")
    fig.update_layout(**CHART_LAYOUT, barmode="stack", height=330, bargap=0.35,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None))
    fig.update_yaxes(range=[0, 100], ticksuffix="%", gridcolor=LINE, title=None)
    fig.update_xaxes(title=None)
    return fig


def view_detail(a: dict, result: rk.RankingResult) -> None:
    post_img = decode(a["post_bytes"])
    pre_img = decode(a["pre_bytes"]) if a.get("pre_bytes") else None
    stats = rk.overall_statistics(result.overall_counts)
    if ss.zone not in [z.letter for z in result.zones]:
        ss.zone = result.zones[0].letter

    header(a, "Detailed damage analysis")

    # --- imagery ------------------------------------------------------------
    with st.container(key="card_imagery"):
        c1, c2 = st.columns([1.6, 1], vertical_alignment="center")
        c1.markdown(damage_legend(), unsafe_allow_html=True)
        ss.alpha = c2.slider("Damage overlay opacity", 0.0, 1.0, float(ss.alpha), 0.05)
        overlay = overlay_mask(post_img, a["mask"], alpha=ss.alpha)
        i1, i2 = st.columns(2, gap="medium")
        if pre_img is not None:
            i1.image(pre_img, caption="Before the disaster (for comparison, not analyzed)")
        else:
            i1.image(post_img, caption="After the disaster (no before image provided)")
        i2.image(overlay, caption="After the disaster, with AI-detected building damage")

    st.write("")

    # --- damage breakdown ----------------------------------------------------
    b1, b2 = st.columns([1, 1.9], gap="large")
    with b1:
        with st.container(key="card_overall"):
            st.markdown('<div class="label-sm">Damage across the image</div>', unsafe_allow_html=True)
            if stats["building_pixels"]:
                shares = {1: stats["no_damage_share"], 2: stats["minor_share"], 3: stats["major_share"],
                          4: stats["destroyed_share"]}
                st.markdown(composition(shares) +
                            f'<div class="muted" style="margin-top:10px">Shares of the detected building area. '
                            f'Buildings cover about {pct(stats["building_coverage"])} of the image.</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="muted">No buildings were detected in this image.</div>',
                            unsafe_allow_html=True)
    with b2:
        with st.container(key="card_zonechart"):
            st.markdown('<div class="label-sm">Damage by zone</div>', unsafe_allow_html=True)
            st.plotly_chart(zone_chart(result), config=CHART_CONFIG)

    st.write("")

    # --- zone inspection -----------------------------------------------------
    with st.container(key="card_inspect"):
        letters = [z.letter for z in result.zones]
        h1, h2 = st.columns([2, 1], vertical_alignment="center")
        h1.markdown('<div class="label-sm">Zone close-up</div>', unsafe_allow_html=True)
        choice = h2.selectbox(
            "Zone", letters, index=letters.index(ss.zone), label_visibility="collapsed",
            format_func=lambda l: f"{result.by_name(l).name} · {PRIORITY_LABEL[result.by_name(l).priority]}",
        )
        if choice != ss.zone:
            ss.zone = choice
        z = result.by_name(ss.zone)

        cols = st.columns([1, 1, 0.95] if pre_img is not None else [1.3, 1], gap="medium")
        if pre_img is not None:
            sx, sy = pre_img.size[0] / post_img.size[0], pre_img.size[1] / post_img.size[1]
            x0, y0, x1, y1 = z.bbox
            pre_box = (round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy))
            cols[0].image(enlarge(crop_box(pre_img, pre_box)), caption=f"{z.name} before")
        cols[-2].image(enlarge(crop_box(overlay, z.bbox)), caption=f"{z.name} after, with detected damage")
        with cols[-1]:
            rows = ""
            if z.score is not None:
                rows = (f'<div class="crow"><span>Severity</span><b>{z.score:.1f} / 5</b></div>'
                        f'<div class="crow"><span>Rank</span><b>{z.rank} of {sum(1 for q in result.zones if q.rank)}'
                        f'</b></div>'
                        f'<div class="crow"><span class="tip" title="Share of the zone the AI classified as '
                        f'building, at any damage level">Building area in zone</span>'
                        f'<b>{pct(z.building_coverage)}</b></div>'
                        f'<div class="crow"><span>Visible damage</span><b>{pct(z.damaged_share)}</b></div>'
                        f'<div class="crow"><span>Major damage or destroyed</span><b>{pct(z.severe_share)}</b></div>')
            st.markdown(
                f'<div style="font-size:1.25rem;font-weight:700">{z.name}&nbsp; {pill(z.priority)}</div>'
                f'<div style="font-size:.92rem;line-height:1.5;margin:8px 0 10px">{esc(rk.zone_insight(z))}</div>'
                + rows
                + (('<div class="label-sm" style="margin-top:14px">Damage mix in this zone</div>'
                    + composition(zone_shares(z))) if z.score is not None else ""),
                unsafe_allow_html=True,
            )

    st.write("")
    technical_details(a, result)


def technical_details(a: dict, result: rk.RankingResult) -> None:
    with st.expander("Technical details"):
        t1, t2 = st.columns(2, gap="large")
        with t1:
            st.markdown("**Processing**")
            st.markdown(
                f"- Processed on {a['device'].upper()} in {a['seconds']:.2f} s\n"
                f"- Image size {a['size'][0]} × {a['size'][1]} px; AI input "
                f"{a['input_size'][0]} × {a['input_size'][1]} px (result resized back to full size)\n"
                f"- Model: `{MODEL_REPO}` / `{MODEL_CHECKPOINT}` (CC BY-NC-SA 4.0)\n"
                f"- The model analyzes the after-disaster image only; it does not perform before/after "
                f"change detection."
            )
        with t2:
            st.markdown("**Scoring method**")
            st.markdown(
                "- Severity = (1 × minor + 3 × major + 5 × destroyed) ÷ detected building area, per zone\n"
                "- Background is excluded; zones without detected buildings are not scored\n"
                f"- Priority levels in use: high ≥ {result.thresholds['high']:g}, "
                f"medium ≥ {result.thresholds['medium']:g} (change under Advanced settings)\n"
                "- Zones are a 3 × 3 division of the image, not geographic areas"
            )

        if a.get("labels_path"):
            st.markdown("**Accuracy check against reference annotations**")
            st.caption("Compares the AI result with the dataset's human annotations for this image. The "
                       "annotations are used only for this check, never as results.")
            if st.toggle("Compute accuracy metrics"):
                from evaluation import evaluate_prediction

                ev = evaluate_prediction(a["mask"], dl.xbd_buildings(dl.load_json(a["labels_path"])))

                def f(v):
                    return "—" if math.isnan(v) else f"{100 * v:.1f}%"

                e1, e2, e3 = st.columns(3)
                e1.metric("Building detection IoU", f(ev["building_iou"]))
                e2.metric("Mean damage-class IoU", f(ev["mean_damage_iou"]))
                e3.metric("Pixel accuracy", f(ev["pixel_accuracy"]))
                st.dataframe(pd.DataFrame([
                    {"Class": m["class"], "IoU": f(m["iou"]), "Precision": f(m["precision"]),
                     "Recall": f(m["recall"])} for m in ev["per_class"].values()
                ]), hide_index=True)
                names = [CLASS_NAMES[c] for c in range(5)]
                fig = go.Figure(go.Heatmap(z=ev["confusion_matrix"], x=names, y=names,
                                           colorscale=[[0, SURFACE], [1, INK]], showscale=False,
                                           hovertemplate="Reference %{y} → AI %{x}: %{z:,} px<extra></extra>"))
                fig.update_layout(**CHART_LAYOUT, height=340)
                fig.update_xaxes(title="AI result")
                fig.update_yaxes(title="Reference annotation", autorange="reversed")
                st.plotly_chart(fig, config=CHART_CONFIG)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if ss.view == "new":
    view_new()
else:
    assessment = ss.assessment
    ranking = rk.rank_zones(assessment["mask"], thresholds)
    if ss.view == "overview":
        view_overview(assessment, ranking)
    else:
        view_detail(assessment, ranking)

st.markdown('<div class="footer">AI-assisted damage assessment for decision support. Results are estimates '
            'from satellite imagery, not verified field assessments. Percentages describe detected building '
            'area, not numbers of buildings.</div>', unsafe_allow_html=True)
