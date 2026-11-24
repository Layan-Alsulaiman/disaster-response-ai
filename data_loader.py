"""Discover local disaster cases and read optional xBD annotation files.

A case is a folder under data/samples/ containing a post-disaster image and,
optionally, the matching pre-disaster image, the xBD label JSON and a
`case.json` with display metadata:

    data/samples/<case>/
        *_pre_disaster.png      (optional, display only)
        *_post_disaster.png     (required, model input)
        *_post_disaster.json    (optional, xBD ground truth - evaluation only)
        case.json               (optional: {"name", "disaster_type", "description"})

xBD image pairs placed directly in data/samples/ (without a sub-folder) are
grouped by their common prefix as well.

Annotation files are ground truth. They are never used as AI predictions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent / "data" / "samples"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# xBD `disaster_type` values -> display names
DISASTER_TYPES = {
    "flooding": "Flood",
    "wind": "Hurricane / Wind",
    "fire": "Wildfire",
    "earthquake": "Earthquake",
    "tsunami": "Tsunami",
    "volcano": "Volcano",
}
UPLOAD_DISASTER_TYPES = ["Earthquake", "Flood", "Hurricane / Wind", "Wildfire", "Tsunami", "Volcano", "Other"]


@dataclass
class DisasterCase:
    case_id: str
    name: str
    disaster_type: str
    post_path: Path
    pre_path: Path | None = None
    labels_path: Path | None = None
    description: str = ""
    capture_date: str = ""

    @property
    def has_labels(self) -> bool:
        return self.labels_path is not None and self.labels_path.is_file()


def _tokens(stem: str) -> set[str]:
    return set(re.split(r"[_\-.\s]+", stem.lower()))


def _role(path: Path) -> str | None:
    """'pre' / 'post' from a filename such as guatemala-volcano_00000003_post_disaster.png."""
    t = _tokens(path.stem)
    if "post" in t:
        return "post"
    if "pre" in t:
        return "pre"
    return None


def _prettify(slug: str) -> str:
    return re.sub(r"[_\-]+", " ", slug).strip().title()


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_case(case_id: str, files: list[Path], folder: Path | None) -> DisasterCase | None:
    images = [p for p in files if p.suffix.lower() in IMAGE_EXTS]
    posts = sorted(p for p in images if _role(p) == "post")
    if not posts:
        return None
    post = posts[0]
    pres = sorted(p for p in images if _role(p) == "pre")
    labels = sorted(p for p in files if p.suffix.lower() == ".json" and _role(p) == "post")

    name, dtype, desc, date = _prettify(case_id), "Unknown", "", ""

    meta = {}
    if labels:
        try:
            meta = load_json(labels[0]).get("metadata", {}) or {}
        except (OSError, ValueError):
            meta = {}
    if meta.get("disaster"):
        name = f"{_prettify(meta['disaster'])} — {case_id.split('_')[-1]}" if "_" in case_id else _prettify(meta["disaster"])
    if meta.get("disaster_type"):
        dtype = DISASTER_TYPES.get(meta["disaster_type"], _prettify(meta["disaster_type"]))
    date = str(meta.get("capture_date", ""))[:10]

    case_json = folder / "case.json" if folder else None
    if case_json and case_json.is_file():
        try:
            cj = load_json(case_json)
            name = cj.get("name", name)
            dtype = cj.get("disaster_type", dtype)
            desc = cj.get("description", desc)
        except (OSError, ValueError):
            pass

    return DisasterCase(
        case_id=case_id, name=name, disaster_type=dtype, post_path=post,
        pre_path=pres[0] if pres else None, labels_path=labels[0] if labels else None,
        description=desc, capture_date=date,
    )


def _xbd_prefix(path: Path) -> str:
    return re.sub(r"_(pre|post)_disaster$", "", path.stem, flags=re.IGNORECASE)


def discover_cases(root: Path | str = SAMPLES_DIR) -> list[DisasterCase]:
    """List all usable cases under `root` (sorted by name). Missing folder -> []."""
    root = Path(root)
    if not root.is_dir():
        return []
    cases = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        case = _build_case(sub.name, [p for p in sub.iterdir() if p.is_file()], sub)
        if case:
            cases.append(case)

    loose: dict[str, list[Path]] = {}
    for p in root.iterdir():
        if p.is_file() and _role(p) and (p.suffix.lower() in IMAGE_EXTS or p.suffix.lower() == ".json"):
            loose.setdefault(_xbd_prefix(p), []).append(p)
    for prefix, files in sorted(loose.items()):
        case = _build_case(prefix, files, None)
        if case:
            cases.append(case)
    return sorted(cases, key=lambda c: c.name)


# ---------------------------------------------------------------------------
# xBD label parsing (ground truth; evaluation and metadata only)
# ---------------------------------------------------------------------------

_POLY_RE = re.compile(r"POLYGON\s*\(\((.+?)\)", re.IGNORECASE)


def parse_wkt_polygon(wkt: str) -> list[tuple[float, float]]:
    """Outer ring of a WKT POLYGON as a list of (x, y)."""
    m = _POLY_RE.search(wkt or "")
    if not m:
        return []
    pts = []
    for pair in m.group(1).split(","):
        parts = pair.strip().split()
        if len(parts) >= 2:
            pts.append((float(parts[0]), float(parts[1])))
    return pts


def xbd_buildings(labels: dict) -> list[tuple[list[tuple[float, float]], str]]:
    """(pixel polygon, damage subtype) for each annotated building in an xBD label dict."""
    out = []
    for feat in labels.get("features", {}).get("xy", []):
        props = feat.get("properties", {})
        if props.get("feature_type", "building") != "building":
            continue
        poly = parse_wkt_polygon(feat.get("wkt", ""))
        if len(poly) >= 3:
            out.append((poly, props.get("subtype", "no-damage")))
    return out


def tile_location(labels: dict) -> tuple[float, float] | None:
    """Mean (lat, lon) of annotated building footprints from xBD `lng_lat` features.

    Only real coordinates from the annotation file are used; None if absent.
    """
    lons, lats = [], []
    for feat in labels.get("features", {}).get("lng_lat", []):
        for x, y in parse_wkt_polygon(feat.get("wkt", "")):
            lons.append(x)
            lats.append(y)
    if not lons:
        return None
    return sum(lats) / len(lats), sum(lons) / len(lons)
