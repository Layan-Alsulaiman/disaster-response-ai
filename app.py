"""Disaster Damage Intelligence: Streamlit front end (dark command-center theme).

Three views, switched from the navigation bar under the header:
  * New Assessment       choose a case or upload imagery, run the AI assessment
  * Response Overview    key indicators, interactive priority map, ranked zones, findings, exports
  * Detailed Analysis    tabs: before/after imagery, zone close-up, damage breakdown, technical details

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
from pathlib import Path

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
LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo.svg"

st.set_page_config(page_title="Disaster Damage Intelligence", page_icon=":material/satellite_alt:",
                   layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

BG, SURFACE, RAISED = "#0F1420", "#192131", "#1F2940"
TEXT, MUTED, BORDER = "#F1F5F9", "#94A3B8", "rgba(148,163,184,.14)"
CORAL, AMBER, TEAL, SLATE = "#F06464", "#E9B44C", "#45B8A0", "#64748B"
TRI = f"linear-gradient(90deg, {CORAL} 0%, {AMBER} 50%, {TEAL} 100%)"


def hexc(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


PRIORITY_HEX = {p: hexc(c) for p, c in rk.PRIORITY_COLORS.items()}
PRIORITY_TEXT = {rk.HIGH: CORAL, rk.MEDIUM: AMBER, rk.LOW: TEAL, rk.INSUFFICIENT: "#A5B1C2"}
PRIORITY_LABEL = {rk.HIGH: "High priority", rk.MEDIUM: "Medium priority", rk.LOW: "Low priority",
                  rk.INSUFFICIENT: "No buildings detected"}
CLASS_HEX = {c: hexc(CLASS_COLORS[c]) for c in range(1, 5)}
DAMAGE_LABEL = {1: "No visible damage", 2: "Minor damage", 3: "Major damage", 4: "Destroyed"}

VIEWS = [("new", "New Assessment", ":material/add_photo_alternate:"),
         ("overview", "Response Overview", ":material/radar:"),
         ("detail", "Detailed Analysis", ":material/query_stats:")]
MAP_LAYERS = ["Priority zones", "Damage detection", "Satellite only"]

CHART_LAYOUT = dict(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, size=13), margin=dict(l=8, r=8, t=8, b=8),
    hoverlabel=dict(bgcolor=SURFACE, bordercolor="#334155", font=dict(color=TEXT, size=13)),
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
    f"""
<style>
  #MainMenu, footer {{visibility: hidden;}}
  header[data-testid="stHeader"] {{background: transparent;}}
  .stApp {{background:
      radial-gradient(1100px 520px at 0% -8%, rgba(240,100,100,.07), transparent 60%),
      radial-gradient(900px 480px at 100% -6%, rgba(69,184,160,.07), transparent 60%),
      {BG};}}
  .block-container {{padding-top: 1.1rem; padding-bottom: 1.2rem; max-width: 1600px;}}

  /* ---------- header ---------- */
  .hero {{position: relative; overflow: hidden; display: flex; justify-content: space-between; align-items: center;
          gap: 24px; padding: 16px 24px 18px 18px; border-radius: 16px; border: 1px solid {BORDER};
          background: linear-gradient(115deg, #1C2640 0%, #172033 48%, #2A1A27 100%);
          box-shadow: 0 18px 40px -28px rgba(0,0,0,.8);}}
  .hero::after {{content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 3px; background: {TRI};}}
  .brand {{display: flex; align-items: center; gap: 16px; min-width: 0;}}
  .brand img {{width: 54px; height: 54px; flex: none; border-radius: 15px;
               filter: drop-shadow(0 6px 16px rgba(124,107,224,.35)) drop-shadow(0 0 10px rgba(240,100,100,.15));}}
  .hero h1 {{font-size: 1.65rem; font-weight: 800; letter-spacing: -.015em; color: {TEXT}; margin: 0; padding: 0;
             line-height: 1.1;}}
  .hero .tagline {{font-size: .7rem; font-weight: 700; letter-spacing: .2em; text-transform: uppercase;
                   margin-top: 6px; background: {TRI}; -webkit-background-clip: text; background-clip: text;
                   color: transparent; width: fit-content;}}
  .hero .case {{text-align: right; min-width: 0;}}
  .hero .case .name {{font-size: 1.12rem; font-weight: 700; color: {TEXT}; margin-bottom: 6px;
                      white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 560px;}}
  .chip-s {{display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; margin-left: 6px;
            border-radius: 999px; background: rgba(148,163,184,.10); border: 1px solid rgba(148,163,184,.18);
            font-size: .76rem; color: #CBD5E1; white-space: nowrap;}}
  .dot {{width: 7px; height: 7px; border-radius: 50%; display: inline-block;}}

  /* ---------- navigation ---------- */
  [class*="st-key-nav_"] button {{background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
      color: {MUTED}; font-weight: 600; min-height: 44px; transition: all .15s ease;}}
  [class*="st-key-nav_"] button:hover {{color: {TEXT}; border-color: rgba(148,163,184,.35); background: {RAISED};}}
  [class*="st-key-nav_"] button:disabled {{opacity: .4;}}

  /* ---------- typography ---------- */
  .label-sm {{font-size: .72rem; font-weight: 700; color: {MUTED}; text-transform: uppercase; letter-spacing: .1em;
              margin: 0 0 10px; display: flex; align-items: center;}}
  .label-sm::before {{content: ""; width: 3px; height: 13px; border-radius: 2px; margin-right: 9px;
                      background: var(--acc, {CORAL});}}
  .muted {{color: {MUTED}; font-size: .84rem;}}
  .tip {{border-bottom: 1px dotted #64748B; cursor: help;}}
  .num {{font-variant-numeric: tabular-nums;}}

  /* ---------- cards ---------- */
  [class*="st-key-card_"], .card {{background: linear-gradient(180deg, #1C2538 0%, {SURFACE} 100%);
      border: 1px solid {BORDER}; border-radius: 14px; padding: 16px 18px;
      box-shadow: 0 14px 34px -24px rgba(0,0,0,.85), inset 0 1px 0 rgba(255,255,255,.03);}}

  .kpi {{position: relative; overflow: hidden; background: linear-gradient(160deg, #1F2A42 0%, {SURFACE} 70%);
         border: 1px solid {BORDER}; border-radius: 14px; padding: 14px 18px 14px; min-height: 96px;}}
  .kpi::before {{content: ""; position: absolute; left: 0; right: 0; top: 0; height: 3px; background: var(--acc);}}
  .kpi::after {{content: ""; position: absolute; right: -40px; top: -40px; width: 130px; height: 130px;
                background: radial-gradient(circle, var(--glow), transparent 70%);}}
  .kpi .k-label {{font-size: .7rem; font-weight: 700; color: {MUTED}; text-transform: uppercase; letter-spacing: .1em;}}
  .kpi .k-row {{display: flex; justify-content: space-between; align-items: flex-end; gap: 10px;}}
  .kpi .k-value {{font-size: 2rem; font-weight: 800; color: {TEXT}; line-height: 1.2; font-variant-numeric: tabular-nums;}}
  .kpi .k-value small {{font-size: .95rem; color: {MUTED}; font-weight: 600; margin-left: 4px;}}
  .kpi .k-sub {{font-size: .8rem; color: {MUTED}; margin-top: 2px;}}
  .mini-grid {{display: grid; grid-template-columns: repeat(3, 12px); gap: 3px; margin-bottom: 4px;}}
  .mini-grid i {{width: 12px; height: 12px; border-radius: 3px; display: block;}}
  .meter {{height: 6px; border-radius: 4px; background: rgba(148,163,184,.15); overflow: hidden; margin-top: 8px;}}
  .meter div {{height: 100%; border-radius: 4px;}}

  /* ---------- pills, legends, bars ---------- */
  .pill {{display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .76rem; font-weight: 700;
          vertical-align: middle; letter-spacing: .01em;}}
  .chip {{display: inline-flex; align-items: center; margin: 4px 16px 0 0; font-size: .8rem; color: #CBD5E1;}}
  .chip i {{width: 10px; height: 10px; border-radius: 3px; margin-right: 7px; display: inline-block;}}
  .cbar {{display: flex; height: 10px; border-radius: 6px; overflow: hidden; background: rgba(148,163,184,.15);
          margin: 6px 0 4px;}}
  .cbar div {{height: 100%;}}
  .crow {{display: flex; justify-content: space-between; font-size: .86rem; padding: 6px 0; color: #CBD5E1;
          border-bottom: 1px solid rgba(148,163,184,.08);}}
  .crow:last-child {{border-bottom: none;}}
  .crow b {{color: {TEXT}; font-variant-numeric: tabular-nums;}}
  .crow span i {{width: 9px; height: 9px; border-radius: 2px; display: inline-block; margin-right: 8px;}}

  /* ---------- buttons ---------- */
  .stButton button, .stDownloadButton button {{border-radius: 10px; font-weight: 600;}}
  .st-key-run button {{width: 100%; min-height: 52px; border: none; color: #10151F; font-weight: 800;
      font-size: 1.02rem; letter-spacing: .01em; background: linear-gradient(135deg, {CORAL} 0%, {AMBER} 100%);
      box-shadow: 0 12px 30px -12px rgba(240,100,100,.65);}}
  .st-key-run button:hover {{filter: brightness(1.07); color: #10151F; border: none;}}
  .st-key-run button:disabled {{background: {RAISED}; color: {MUTED}; box-shadow: none;}}
  [class*="st-key-zbtn_"] button {{justify-content: flex-start; text-align: left; background: #1B2436;
      border: 1px solid {BORDER}; border-left: 4px solid {SLATE}; border-radius: 10px; padding: 7px 12px;
      min-height: 0; color: #E2E8F0;}}
  [class*="st-key-zbtn_"] button:hover {{background: {RAISED}; color: {TEXT};}}
  [class*="st-key-zbtn_"] button p {{font-size: .9rem;}}
  [class*="st-key-dl_"] button, .st-key-inspect button {{width: 100%; background: {RAISED};
      border: 1px solid rgba(148,163,184,.22);}}

  /* ---------- tabs ---------- */
  .stTabs [data-baseweb="tab-list"] {{gap: 6px; border-bottom: 1px solid {BORDER};}}
  .stTabs [data-baseweb="tab"] {{background: {SURFACE}; border: 1px solid {BORDER}; border-bottom: none;
      border-radius: 10px 10px 0 0; padding: 8px 18px; color: {MUTED}; font-weight: 600;}}
  .stTabs [aria-selected="true"] {{background: {RAISED}; color: {TEXT};}}
  .stTabs [data-baseweb="tab-highlight"] {{background: {TRI}; height: 3px;}}
  .stTabs [data-baseweb="tab-border"] {{display: none;}}

  /* ---------- imagery ---------- */
  [data-testid="stImage"] img {{border-radius: 12px; border: 1px solid {BORDER};}}
  .frame-label {{font-size: .7rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
                 color: {MUTED}; margin: 0 0 6px; display: flex; align-items: center; gap: 8px;}}
  .placeholder {{background: repeating-linear-gradient(45deg, #151C2B, #151C2B 10px, #182032 10px, #182032 20px);
                 border: 1px dashed rgba(148,163,184,.25); border-radius: 12px; height: 330px; display: flex;
                 align-items: center; justify-content: center; text-align: center; color: {MUTED};
                 font-size: .85rem; padding: 16px;}}
  .footer {{position: relative; color: #7C8AA0; font-size: .76rem; margin-top: 22px; padding-top: 12px;}}
  .footer::before {{content: ""; position: absolute; left: 0; top: 0; width: 120px; height: 2px; background: {TRI};
                    border-radius: 2px;}}
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


def label(text: str, acc: str = CORAL) -> str:
    return f'<div class="label-sm" style="--acc:{acc}">{text}</div>'


def pill(priority: str) -> str:
    c = PRIORITY_HEX[priority]
    return (f'<span class="pill" style="background:{rgba(c, .15)};color:{PRIORITY_TEXT[priority]};'
            f'border:1px solid {rgba(c, .4)}">{PRIORITY_LABEL[priority]}</span>')


def kpi(lab: str, value: str, sub: str, acc: str, tip: str = "", side: str = "", extra: str = "") -> str:
    lab_html = f'<span class="tip" title="{esc(tip)}">{lab}</span>' if tip else lab
    return (f'<div class="kpi" style="--acc:{acc};--glow:{rgba(acc, .16)}">'
            f'<div class="k-label">{lab_html}</div>'
            f'<div class="k-row"><div><div class="k-value">{value}</div>'
            + (f'<div class="k-sub">{sub}</div>' if sub else "")
            + f'</div>{side}</div>{extra}</div>')


def damage_legend() -> str:
    return "".join(f'<span class="chip"><i style="background:{CLASS_HEX[c]}"></i>{DAMAGE_LABEL[c]}</span>'
                   for c in range(1, 5))


def priority_legend() -> str:
    return "".join(f'<span class="chip"><i style="background:{PRIORITY_HEX[p]}"></i>{PRIORITY_LABEL[p]}</span>'
                   for p in rk.PRIORITY_ORDER)


def composition(shares: dict[int, float], rows: bool = True) -> str:
    """Stacked bar of damage classes (shares of building area), optionally with a labelled breakdown."""
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


@st.cache_data(show_spinner=False)
def logo_img() -> str:
    try:
        data = base64.b64encode(LOGO_PATH.read_bytes()).decode()
    except OSError:
        return ""
    return f'<img src="data:image/svg+xml;base64,{data}" alt="">'


def to_data_uri(img: Image.Image, max_side: int = 1200) -> str:
    img = img.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def enlarge(img: Image.Image, target: int = 900) -> Image.Image:
    """Upscale small crops so they fill their column (st.image never upscales)."""
    w, h = img.size
    scale = target / max(w, h)
    return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS) if scale > 1 else img


def zone_label(z) -> str:
    return "No buildings" if z.score is None else f"{z.priority} · {z.score:.1f}"


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
# Header and navigation
# ---------------------------------------------------------------------------

has_result = ss.assessment is not None
if has_result:
    a0 = ss.assessment
    chips = f'<span class="chip-s"><span class="dot" style="background:{AMBER}"></span>{esc(a0["type"])}</span>'
    if a0.get("date"):
        chips += f'<span class="chip-s">Captured {esc(a0["date"])}</span>'
    chips += f'<span class="chip-s"><span class="dot" style="background:{TEAL}"></span>AI analysis complete</span>'
    case_html = (f'<div class="case"><div class="name">{esc(a0["name"])}</div>{chips}</div>')
else:
    case_html = ('<div class="case"><span class="chip-s"><span class="dot" style="background:#64748B"></span>'
                 'No active assessment</span></div>')

st.markdown(
    f'<div class="hero"><div class="brand">{logo_img()}<div><h1>Disaster Damage Intelligence</h1>'
    f'<div class="tagline">Satellite assessment · Response prioritization</div></div></div>'
    f'{case_html}</div>',
    unsafe_allow_html=True,
)

st.write("")
nav_cols = st.columns([1, 1, 1, 0.12, 0.85], vertical_alignment="center")
for col, (key, lab, icon) in zip(nav_cols[:3], VIEWS):
    col.button(lab, key=f"nav_{key}", icon=icon, on_click=go_to, args=(key,),
               disabled=key != "new" and not has_result)
css(f""".st-key-nav_{ss.view} button {{color: {TEXT}; border-color: {rgba(CORAL, .55)};
     background: linear-gradient(135deg, {rgba(CORAL, .20)} 0%, {rgba(AMBER, .10)} 100%);
     box-shadow: 0 8px 22px -12px {rgba(CORAL, .7)};}}""")

with nav_cols[4]:
    with st.popover("Advanced settings", icon=":material/tune:"):
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

if thresholds_note:
    st.warning(thresholds_note, icon=":material/warning:")
if ss.pop("_toast", None):
    st.toast("Assessment complete", icon=":material/check_circle:")
st.write("")


# ---------------------------------------------------------------------------
# View 1: New Assessment
# ---------------------------------------------------------------------------


def view_new() -> None:
    cases = dl.discover_cases()
    left, right = st.columns([1, 1.45], gap="large")

    pre_bytes = post_bytes = None
    labels_path = None
    capture_date = ""
    with left:
        with st.container(key="card_setup"):
            st.markdown(label("Imagery source", CORAL), unsafe_allow_html=True)
            options = ["Existing case", "Upload images"] if cases else ["Upload images"]
            source = st.segmented_control("Imagery source", options, default=options[0],
                                          label_visibility="collapsed", key="source") or options[0]

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
                    st.caption("No saved cases found in `data/samples/`.")
                post_file = st.file_uploader("After-disaster image",
                                             type=["png", "jpg", "jpeg", "tif", "tiff"])
                pre_file = st.file_uploader("Before-disaster image (optional)",
                                            type=["png", "jpg", "jpeg", "tif", "tiff"])
                post_bytes = post_file.getvalue() if post_file else None
                pre_bytes = pre_file.getvalue() if pre_file else None
                default_name, default_type, key_suffix = "", "Other", "upload"

            st.markdown(label("Disaster details", AMBER), unsafe_allow_html=True)
            types = list(dl.UPLOAD_DISASTER_TYPES)
            if default_type not in types:
                default_type = "Other"
            n1, n2 = st.columns([1.5, 1])
            name = n1.text_input("Disaster name", default_name, key=f"name_{key_suffix}",
                                 placeholder="e.g. Hurricane Michael, Panama City")
            dtype = n2.selectbox("Disaster type", types, index=types.index(default_type), key=f"type_{key_suffix}")

    post_img = pre_img = None
    for raw, which in ((post_bytes, "after"), (pre_bytes, "before")):
        if raw is None:
            continue
        try:
            img = decode(raw)
        except Exception as exc:
            show_error(f"The {which}-disaster image could not be read.", "Please use a PNG, JPEG or TIFF image.",
                       exc)
            img = None
        if which == "after":
            post_img = img
        else:
            pre_img = img

    with left:
        st.write("")
        run = st.button("Analyze with AI", type="primary", icon=":material/bolt:", disabled=post_img is None,
                        key="run")

    with right:
        with st.container(key="card_preview"):
            st.markdown(label("Imagery preview", TEAL), unsafe_allow_html=True)
            p1, p2 = st.columns(2, gap="medium")
            with p1:
                st.markdown('<div class="frame-label"><span class="dot" style="background:#64748B"></span>'
                            'Before · reference</div>', unsafe_allow_html=True)
                if pre_img is not None:
                    st.image(pre_img)
                else:
                    st.markdown('<div class="placeholder">No before-disaster image<br>(optional)</div>',
                                unsafe_allow_html=True)
            with p2:
                st.markdown(f'<div class="frame-label"><span class="dot" style="background:{CORAL}"></span>'
                            f'After · analyzed by AI</div>', unsafe_allow_html=True)
                if post_img is not None:
                    st.image(post_img)
                else:
                    st.markdown('<div class="placeholder">Select a case or upload an<br>after-disaster image</div>',
                                unsafe_allow_html=True)

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
# Priority map
# ---------------------------------------------------------------------------


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


def map_base(post_img: Image.Image, mask: np.ndarray, result: rk.RankingResult, selected: str, layer: str):
    grid = result.grid_order()
    if layer == "Damage detection":
        img, tint = overlay_mask(post_img, mask, alpha=0.6), 0.0
    elif layer == "Satellite only":
        img, tint = post_img, 0.0
    else:
        img, tint = post_img, 0.22
    return draw_zone_grid(img, grid, rk.PRIORITY_COLORS, selected=selected, fill_alpha=tint, label=zone_label)


def priority_map_figure(base: Image.Image, result: rk.RankingResult, height: int = 610) -> go.Figure:
    grid = result.grid_order()
    w, h = base.size
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
            marker=dict(size=46, color=TEXT, opacity=0), name=z.name, showlegend=False,
            selected=dict(marker=dict(opacity=0)), unselected=dict(marker=dict(opacity=0)),
            hovertemplate=f"<b>{z.name}</b><br>{detail}<extra></extra>",
        ))
    fig.add_layout_image(dict(source=to_data_uri(base), xref="x", yref="y", x=0, y=0, sizex=w, sizey=h,
                              sizing="stretch", layer="below"))
    fig.update_xaxes(visible=False, range=[0, w], fixedrange=True, constrain="domain")
    fig.update_yaxes(visible=False, range=[h, 0], fixedrange=True, scaleanchor="x", scaleratio=1,
                     constrain="domain")
    fig.update_layout(**{**CHART_LAYOUT, "margin": dict(l=0, r=0, t=0, b=0)}, height=height, dragmode=False,
                      hovermode="closest", clickmode="event+select")
    return fig


# ---------------------------------------------------------------------------
# View 2: Response Overview
# ---------------------------------------------------------------------------


def kpi_row(result: rk.RankingResult, stats: dict, scored: list) -> None:
    top = scored[0] if scored else None
    severe = stats["major_share"] + stats["destroyed_share"]
    mini = "".join(f'<i style="background:{PRIORITY_HEX[z.priority]};opacity:{1 if z.score is not None else .45}" '
                   f'title="{z.name}: {PRIORITY_LABEL[z.priority]}"></i>' for z in result.grid_order())
    shares = {1: stats["no_damage_share"], 2: stats["minor_share"], 3: stats["major_share"],
              4: stats["destroyed_share"]}
    k = st.columns(4, gap="medium")
    k[0].markdown(kpi("High-priority zones", f'{result.count(rk.HIGH)}<small>/ {len(result.zones)}</small>',
                      f"{result.count(rk.MEDIUM)} medium · {result.count(rk.LOW)} low", CORAL,
                      "Zones whose severity reached the high-priority level.",
                      side=f'<div class="mini-grid">{mini}</div>'), unsafe_allow_html=True)
    k[1].markdown(kpi("Damaged building area", pct(stats["damaged_share"]) if scored else "—",
                      "", AMBER,
                      "Share of the building area detected by the AI that shows minor damage, major damage or "
                      "destruction. This is an area share, not a count of buildings.",
                      extra=composition(shares, rows=False) if scored else ""), unsafe_allow_html=True)
    k[2].markdown(kpi("Severe damage", pct(severe) if scored else "—", "", CORAL,
                      "Share of the detected building area classified as major damage or destroyed.",
                      extra=f'<div class="meter"><div style="width:{100 * severe:.1f}%;background:{CORAL}">'
                            f'</div></div>' if scored else ""), unsafe_allow_html=True)
    acc = PRIORITY_HEX[top.priority] if top else SLATE
    k[3].markdown(kpi("Most affected zone", top.name if top else "—",
                      f"severity {top.score:.1f} / 5" if top else "no buildings detected", acc,
                      "The zone with the highest severity score.",
                      side=pill(top.priority) if top else ""), unsafe_allow_html=True)


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
    zc = PRIORITY_HEX[zone.priority]
    rules.append(f".st-key-zbtn_{zone.letter} button {{background: {rgba(zc, .14)}; border-color: {rgba(zc, .55)};"
                 f"border-left-color: {zc}; color: {TEXT};}}")
    css("\n".join(rules))

    kpi_row(result, stats, scored)
    st.write("")

    map_col, side_col = st.columns([1.75, 1], gap="large")
    with map_col:
        with st.container(key="card_map"):
            h1, h2 = st.columns([1, 1.6], vertical_alignment="center")
            h1.markdown(label("Priority map", CORAL), unsafe_allow_html=True)
            layer = h2.segmented_control("Map layer", MAP_LAYERS, default=MAP_LAYERS[0], key="map_layer",
                                         label_visibility="collapsed") or MAP_LAYERS[0]
            base = map_base(post_img, a["mask"], result, zone.letter, layer)
            st.plotly_chart(priority_map_figure(base, result), config=CHART_CONFIG,
                            on_select="rerun", selection_mode="points", key="priority_map")
            legend = damage_legend() if layer == "Damage detection" else priority_legend()
            st.markdown(f'<div>{legend}</div>', unsafe_allow_html=True)

    with side_col:
        with st.container(key="card_rank"):
            st.markdown(label("Zones by severity", AMBER), unsafe_allow_html=True)
            if not scored:
                st.markdown('<div class="muted">No buildings were detected, so no zone could be ranked.</div>',
                            unsafe_allow_html=True)
            for z in scored:
                st.button(f"**{z.name}**  ·  {z.priority}  ·  {z.score:.1f}", key=f"zbtn_{z.letter}",
                          on_click=select_zone, args=(z.letter,))
            if unscored:
                st.markdown(f'<div class="muted">No buildings detected: '
                            f'{", ".join(z.letter for z in unscored)}</div>', unsafe_allow_html=True)

        st.write("")
        with st.container(key="card_zone"):
            st.markdown(label("Selected zone", zc), unsafe_allow_html=True)
            rank = f"Rank {zone.rank} of {len(scored)}" if zone.rank else "Not ranked"
            sev = (f'<span class="tip" title="0 = no visible damage, 5 = all detected building area destroyed">'
                   f'Severity <b class="num" style="color:{TEXT}">{zone.score:.1f} / 5</b></span> · {rank}'
                   if zone.score is not None else rank)
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<span style="font-size:1.35rem;font-weight:800;color:{TEXT}">{zone.name}</span>'
                f'{pill(zone.priority)}</div>'
                f'<div class="muted" style="margin:4px 0 10px">{sev}</div>'
                f'<div style="font-size:.92rem;line-height:1.5;color:#E2E8F0;margin-bottom:6px">'
                f'{esc(rk.zone_insight(zone))}</div>'
                + (composition(zone_shares(zone), rows=False) if zone.score is not None else ""),
                unsafe_allow_html=True,
            )
            st.button("Inspect zone in detail", key="inspect", icon=":material/zoom_in:", on_click=inspect_zone,
                      args=(zone.letter,))

    st.write("")
    with st.container(key="card_summary"):
        s1, s2 = st.columns([2.3, 1], gap="large")
        with s1:
            st.markdown(label("Key findings", CORAL)
                        + f'<div style="font-size:1.02rem;line-height:1.6;color:{TEXT}">'
                          f'{esc(rk.plain_summary(result))}</div>', unsafe_allow_html=True)
        with s2:
            st.markdown(label("Export", TEAL), unsafe_allow_html=True)
            export_buttons(a, result, post_img)


def export_buttons(a: dict, result: rk.RankingResult, post_img: Image.Image) -> None:
    buf = io.BytesIO()
    overlay_mask(post_img, a["mask"], alpha=ss.alpha).save(buf, format="PNG")
    st.download_button("Damage overlay (PNG)", buf.getvalue(), "damage_overlay.png", "image/png",
                       key="dl_overlay", icon=":material/image:")

    df = pd.DataFrame(rk.zones_table(result)).rename(columns={
        "Score": "Severity (0-5)", "Building px %": "Building area (% of zone)",
        "No Damage %": "No visible damage (% of building area)", "Minor %": "Minor damage (% of building area)",
        "Major %": "Major damage (% of building area)", "Destroyed %": "Destroyed (% of building area)",
    })
    df["Priority"] = df["Priority"].replace({rk.INSUFFICIENT: PRIORITY_LABEL[rk.INSUFFICIENT]})
    st.download_button("Zone ranking (CSV)", df.to_csv(index=False).encode(), "zone_ranking.csv", "text/csv",
                       key="dl_csv", icon=":material/table_chart:")

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
                       "text/plain", key="dl_txt", icon=":material/description:")


# ---------------------------------------------------------------------------
# View 3: Detailed Analysis (tabs)
# ---------------------------------------------------------------------------


def zone_chart(result: rk.RankingResult) -> go.Figure:
    grid = result.grid_order()
    ticks = [f"{z.name}<br><span style='color:{PRIORITY_TEXT[z.priority]}'>"
             f"{'No data' if z.score is None else z.priority}</span>" for z in grid]
    fig = go.Figure()
    for c in range(1, 5):
        fig.add_bar(name=DAMAGE_LABEL[c], x=ticks, y=[100 * z.share(c) for z in grid], marker_color=CLASS_HEX[c],
                    hovertemplate=f"%{{x}}<br>{DAMAGE_LABEL[c]}: %{{y:.0f}}% of building area<extra></extra>")
    fig.update_layout(**CHART_LAYOUT, barmode="stack", height=340, bargap=0.38,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None))
    fig.update_yaxes(range=[0, 100], ticksuffix="%", gridcolor="rgba(148,163,184,.12)", title=None)
    fig.update_xaxes(title=None)
    return fig


def view_detail(a: dict, result: rk.RankingResult) -> None:
    post_img = decode(a["post_bytes"])
    pre_img = decode(a["pre_bytes"]) if a.get("pre_bytes") else None
    stats = rk.overall_statistics(result.overall_counts)
    if ss.zone not in [z.letter for z in result.zones]:
        ss.zone = result.zones[0].letter

    t_img, t_zone, t_break, t_tech = st.tabs(["Before & after", "Zone close-up", "Damage breakdown",
                                              "Technical details"])

    # --- before & after -------------------------------------------------------
    with t_img:
        c1, c2 = st.columns([1.7, 1], vertical_alignment="center")
        c1.markdown(damage_legend(), unsafe_allow_html=True)
        ss.alpha = c2.slider("Damage overlay opacity", 0.0, 1.0, float(ss.alpha), 0.05)
        overlay = overlay_mask(post_img, a["mask"], alpha=ss.alpha)
        i1, i2 = st.columns(2, gap="medium")
        with i1:
            if pre_img is not None:
                st.markdown('<div class="frame-label"><span class="dot" style="background:#64748B"></span>'
                            'Before · reference only</div>', unsafe_allow_html=True)
                st.image(pre_img)
            else:
                st.markdown('<div class="frame-label"><span class="dot" style="background:#64748B"></span>'
                            'After · original (no before image provided)</div>', unsafe_allow_html=True)
                st.image(post_img)
        with i2:
            st.markdown(f'<div class="frame-label"><span class="dot" style="background:{CORAL}"></span>'
                        f'After · AI-detected building damage</div>', unsafe_allow_html=True)
            st.image(overlay)

    # --- zone close-up ----------------------------------------------------------
    with t_zone:
        letters = [z.letter for z in result.zones]
        h1, h2 = st.columns([2, 1], vertical_alignment="center")
        h1.markdown(label("Zone close-up", PRIORITY_HEX[result.by_name(ss.zone).priority]), unsafe_allow_html=True)
        choice = h2.selectbox(
            "Zone", letters, index=letters.index(ss.zone), label_visibility="collapsed",
            format_func=lambda l: f"{result.by_name(l).name} · {PRIORITY_LABEL[result.by_name(l).priority]}",
        )
        if choice != ss.zone:
            ss.zone = choice
        z = result.by_name(ss.zone)

        cols = st.columns([1, 1, 0.9] if pre_img is not None else [1.3, 1], gap="medium")
        if pre_img is not None:
            sx, sy = pre_img.size[0] / post_img.size[0], pre_img.size[1] / post_img.size[1]
            x0, y0, x1, y1 = z.bbox
            pre_box = (round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy))
            with cols[0]:
                st.markdown(f'<div class="frame-label">{z.name} · before</div>', unsafe_allow_html=True)
                st.image(enlarge(crop_box(pre_img, pre_box)))
        with cols[-2]:
            st.markdown(f'<div class="frame-label">{z.name} · after, detected damage</div>', unsafe_allow_html=True)
            st.image(enlarge(crop_box(overlay, z.bbox)))
        with cols[-1]:
            with st.container(key="card_zonestats"):
                rows = ""
                if z.score is not None:
                    rows = (f'<div class="crow"><span>Severity</span><b>{z.score:.1f} / 5</b></div>'
                            f'<div class="crow"><span>Rank</span><b>{z.rank} of '
                            f'{sum(1 for q in result.zones if q.rank)}</b></div>'
                            f'<div class="crow"><span class="tip" title="Share of the zone the AI classified as '
                            f'building, at any damage level">Building area in zone</span>'
                            f'<b>{pct(z.building_coverage)}</b></div>'
                            f'<div class="crow"><span>Visible damage</span><b>{pct(z.damaged_share)}</b></div>'
                            f'<div class="crow"><span>Major damage or destroyed</span><b>{pct(z.severe_share)}</b>'
                            f'</div>')
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:center">'
                    f'<span style="font-size:1.3rem;font-weight:800;color:{TEXT}">{z.name}</span>{pill(z.priority)}'
                    f'</div><div style="font-size:.9rem;line-height:1.5;color:#E2E8F0;margin:8px 0 10px">'
                    f'{esc(rk.zone_insight(z))}</div>' + rows
                    + ((label("Damage mix in this zone", AMBER) + composition(zone_shares(z)))
                       if z.score is not None else ""),
                    unsafe_allow_html=True,
                )

    # --- damage breakdown -------------------------------------------------------
    with t_break:
        b1, b2 = st.columns([1, 2], gap="large")
        with b1:
            with st.container(key="card_overall"):
                st.markdown(label("Damage across the image", CORAL), unsafe_allow_html=True)
                if stats["building_pixels"]:
                    shares = {1: stats["no_damage_share"], 2: stats["minor_share"], 3: stats["major_share"],
                              4: stats["destroyed_share"]}
                    st.markdown(composition(shares) +
                                f'<div class="muted" style="margin-top:10px">Buildings cover about '
                                f'{pct(stats["building_coverage"])} of the image.</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="muted">No buildings were detected in this image.</div>',
                                unsafe_allow_html=True)
        with b2:
            with st.container(key="card_zonechart"):
                st.markdown(label("Damage by zone", AMBER), unsafe_allow_html=True)
                st.plotly_chart(zone_chart(result), config=CHART_CONFIG)

    # --- technical details ------------------------------------------------------
    with t_tech:
        technical_details(a, result)


def technical_details(a: dict, result: rk.RankingResult) -> None:
    t1, t2 = st.columns(2, gap="large")
    with t1:
        with st.container(key="card_processing"):
            st.markdown(label("Processing", TEAL), unsafe_allow_html=True)
            st.markdown(
                f"- Processed on {a['device'].upper()} in {a['seconds']:.2f} s\n"
                f"- Image size {a['size'][0]} × {a['size'][1]} px; AI input "
                f"{a['input_size'][0]} × {a['input_size'][1]} px (result resized back to full size)\n"
                f"- Model: `{MODEL_REPO}` / `{MODEL_CHECKPOINT}` (CC BY-NC-SA 4.0)\n"
                f"- The model analyzes the after-disaster image only; it does not perform before/after "
                f"change detection."
            )
    with t2:
        with st.container(key="card_method"):
            st.markdown(label("Scoring method", AMBER), unsafe_allow_html=True)
            st.markdown(
                "- Severity = (1 × minor + 3 × major + 5 × destroyed) ÷ detected building area, per zone\n"
                "- Background is excluded; zones without detected buildings are not scored\n"
                f"- Priority levels in use: high ≥ {result.thresholds['high']:g}, "
                f"medium ≥ {result.thresholds['medium']:g} (change under Advanced settings)\n"
                "- Zones are a 3 × 3 division of the image, not geographic areas"
            )

    if a.get("labels_path"):
        st.write("")
        with st.container(key="card_accuracy"):
            st.markdown(label("Accuracy check against reference annotations", CORAL), unsafe_allow_html=True)
            st.caption("Reference annotations are used only for this check, never as results.")
            if st.toggle("Compute accuracy metrics"):
                from evaluation import evaluate_prediction

                ev = evaluate_prediction(a["mask"], dl.xbd_buildings(dl.load_json(a["labels_path"])))

                def f(v):
                    return "—" if math.isnan(v) else f"{100 * v:.1f}%"

                e1, e2, e3 = st.columns(3)
                e1.metric("Building detection IoU", f(ev["building_iou"]))
                e2.metric("Mean damage-class IoU", f(ev["mean_damage_iou"]))
                e3.metric("Pixel accuracy", f(ev["pixel_accuracy"]))
                m1, m2 = st.columns([1, 1.2], gap="large")
                m1.dataframe(pd.DataFrame([
                    {"Class": m["class"], "IoU": f(m["iou"]), "Precision": f(m["precision"]),
                     "Recall": f(m["recall"])} for m in ev["per_class"].values()
                ]), hide_index=True)
                names = [CLASS_NAMES[c] for c in range(5)]
                fig = go.Figure(go.Heatmap(z=ev["confusion_matrix"], x=names, y=names,
                                           colorscale=[[0, SURFACE], [1, CORAL]], showscale=False,
                                           hovertemplate="Reference %{y} → AI %{x}: %{z:,} px<extra></extra>"))
                fig.update_layout(**CHART_LAYOUT, height=320)
                fig.update_xaxes(title="AI result")
                fig.update_yaxes(title="Reference annotation", autorange="reversed")
                m2.plotly_chart(fig, config=CHART_CONFIG)


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

st.markdown('<div class="footer">AI-assisted estimates for decision support, not verified field assessments. '
            'Percentages refer to detected building area, not building counts.</div>', unsafe_allow_html=True)
