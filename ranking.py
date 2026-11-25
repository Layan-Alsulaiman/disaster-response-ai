"""Zone division, damage severity scoring and priority ranking.

Everything here works on a predicted class-ID mask (0..4) produced by the model.

Score for one zone (transparent and easy to change via DAMAGE_WEIGHTS):

    score = (1 * minor_px + 3 * major_px + 5 * destroyed_px) / building_px
    building_px = no_damage_px + minor_px + major_px + destroyed_px   (background excluded)

The score ranges from 0 (all building pixels undamaged) to 5 (all destroyed).
Zones with no predicted building pixels get "Insufficient Data", not "Low".

Priority thresholds are preliminary design choices, NOT validated emergency
response standards. This ranks visual damage severity only; it does not
estimate casualties, population exposure or real-world rescue urgency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from image_processing import CLASS_NAMES, NUM_CLASSES

# --- Configuration (edit here or override from the dashboard sidebar) ---
DAMAGE_WEIGHTS = {1: 0.0, 2: 1.0, 3: 3.0, 4: 5.0}   # class ID -> weight
MAX_SCORE = max(DAMAGE_WEIGHTS.values())
DEFAULT_THRESHOLDS = {"high": 2.5, "medium": 1.0}    # score >= high -> High, >= medium -> Medium
GRID_SIZE = 3                                         # 3 x 3 = 9 zones, named A..I row by row

HIGH, MEDIUM, LOW, INSUFFICIENT = "High", "Medium", "Low", "Insufficient Data"
PRIORITY_ORDER = [HIGH, MEDIUM, LOW, INSUFFICIENT]
PRIORITY_COLORS = {                                   # RGB, used by the priority map
    HIGH: (196, 69, 54),            # Emergency Red  #C44536
    MEDIUM: (217, 154, 50),         # Amber          #D99A32
    LOW: (57, 133, 120),            # Muted Teal     #398578
    INSUFFICIENT: (156, 163, 175),  # Neutral grey   #9CA3AF
}
BUILDING_CLASSES = (1, 2, 3, 4)
DAMAGE_CLASSES = (2, 3, 4)


@dataclass
class Zone:
    name: str                       # "Zone A" .. "Zone I"
    row: int
    col: int
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1) in pixels, x1/y1 exclusive
    counts: dict[int, int]          # predicted pixels per class ID 0..4
    score: float | None = None      # None when no building pixels were predicted
    priority: str = INSUFFICIENT
    rank: int | None = None         # 1 = highest score; None for Insufficient Data
    explanation: str = ""

    @property
    def letter(self) -> str:
        return self.name.split()[-1]

    @property
    def total_pixels(self) -> int:
        return sum(self.counts.values())

    @property
    def building_pixels(self) -> int:
        return sum(self.counts[c] for c in BUILDING_CLASSES)

    def share(self, class_id: int) -> float:
        """Fraction of this zone's predicted building pixels in `class_id` (0..1)."""
        b = self.building_pixels
        return self.counts[class_id] / b if b else 0.0

    @property
    def damaged_share(self) -> float:
        return sum(self.share(c) for c in DAMAGE_CLASSES)

    @property
    def severe_share(self) -> float:
        """Major + Destroyed share of building pixels."""
        return self.share(3) + self.share(4)

    @property
    def building_coverage(self) -> float:
        """Fraction of the zone area predicted as building (any damage level)."""
        t = self.total_pixels
        return self.building_pixels / t if t else 0.0


@dataclass
class RankingResult:
    zones: list[Zone]               # sorted: highest score first, Insufficient Data last
    thresholds: dict[str, float]
    overall_counts: dict[int, int] = field(default_factory=dict)

    def by_name(self, name: str) -> Zone:
        for z in self.zones:
            if z.name == name or z.letter == name:
                return z
        raise KeyError(name)

    def grid_order(self) -> list[Zone]:
        """Zones in A..I order (row by row), independent of ranking."""
        return sorted(self.zones, key=lambda z: (z.row, z.col))

    def count(self, priority: str) -> int:
        return sum(1 for z in self.zones if z.priority == priority)


# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------

def validate_thresholds(thresholds: dict[str, float]) -> dict[str, float]:
    high, medium = float(thresholds["high"]), float(thresholds["medium"])
    if not (0.0 <= medium < high <= MAX_SCORE):
        raise ValueError(
            f"Thresholds must satisfy 0 <= medium < high <= {MAX_SCORE}; got medium={medium}, high={high}"
        )
    return {"high": high, "medium": medium}


def split_into_zones(height: int, width: int, grid: int = GRID_SIZE) -> list[tuple[str, int, int, tuple]]:
    """Split an image into grid x grid sectors. Returns (name, row, col, bbox).

    Uses near-equal splits so every pixel belongs to exactly one zone even when
    the size is not divisible by `grid`.
    """
    ys = np.linspace(0, height, grid + 1).round().astype(int)
    xs = np.linspace(0, width, grid + 1).round().astype(int)
    zones = []
    for r in range(grid):
        for c in range(grid):
            letter = chr(ord("A") + r * grid + c)
            zones.append((f"Zone {letter}", r, c, (int(xs[c]), int(ys[r]), int(xs[c + 1]), int(ys[r + 1]))))
    return zones


def count_classes(mask: np.ndarray) -> dict[int, int]:
    counts = np.bincount(mask.ravel().astype(np.int64), minlength=NUM_CLASSES)
    return {c: int(counts[c]) for c in range(NUM_CLASSES)}


def damage_score(counts: dict[int, int], weights: dict[int, float] = DAMAGE_WEIGHTS) -> float | None:
    """Weighted damage score in [0, 5]; None if there are no building pixels."""
    building = sum(counts[c] for c in BUILDING_CLASSES)
    if building == 0:
        return None
    return sum(weights[c] * counts[c] for c in BUILDING_CLASSES) / building


def classify_priority(score: float | None, thresholds: dict[str, float] = DEFAULT_THRESHOLDS) -> str:
    if score is None:
        return INSUFFICIENT
    if score >= thresholds["high"]:
        return HIGH
    if score >= thresholds["medium"]:
        return MEDIUM
    return LOW


def rank_zones(
    mask: np.ndarray,
    thresholds: dict[str, float] | None = None,
    grid: int = GRID_SIZE,
) -> RankingResult:
    """Divide a predicted mask into zones, score, classify and sort them."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {mask.shape}")
    if mask.size and (mask.min() < 0 or mask.max() >= NUM_CLASSES):
        raise ValueError("mask values must be class IDs 0..4")
    thresholds = validate_thresholds(thresholds or DEFAULT_THRESHOLDS)

    h, w = mask.shape
    zones = []
    for name, r, c, (x0, y0, x1, y1) in split_into_zones(h, w, grid):
        counts = count_classes(mask[y0:y1, x0:x1])
        score = damage_score(counts)
        zones.append(Zone(name, r, c, (x0, y0, x1, y1), counts, score, classify_priority(score, thresholds)))

    # Highest score first; ties broken by destroyed share, then major share, then name.
    # Zones without building pixels go last.
    zones.sort(key=lambda z: (
        z.score is None,
        -(z.score or 0.0),
        -z.share(4),
        -z.share(3),
        z.name,
    ))
    rank = 0
    for z in zones:
        if z.score is not None:
            rank += 1
            z.rank = rank
        z.explanation = explain_zone(z, thresholds)

    return RankingResult(zones=zones, thresholds=thresholds, overall_counts=count_classes(mask))


# ---------------------------------------------------------------------------
# Overall statistics
# ---------------------------------------------------------------------------

def overall_statistics(counts: dict[int, int]) -> dict[str, float]:
    """Image-level proportions of predicted building pixels (pixels, not buildings)."""
    total = sum(counts.values())
    building = sum(counts[c] for c in BUILDING_CLASSES)

    def frac(x):
        return x / building if building else 0.0

    return {
        "total_pixels": total,
        "building_pixels": building,
        "building_coverage": building / total if total else 0.0,
        "no_damage_share": frac(counts[1]),
        "minor_share": frac(counts[2]),
        "major_share": frac(counts[3]),
        "destroyed_share": frac(counts[4]),
        "damaged_share": frac(counts[2] + counts[3] + counts[4]),
        "overall_score": damage_score(counts),
    }


# ---------------------------------------------------------------------------
# Template-based explanations (only restate computed numbers)
# ---------------------------------------------------------------------------

def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def explain_zone(z: Zone, thresholds: dict[str, float]) -> str:
    if z.score is None:
        return (f"{z.name} is marked Insufficient Data because the model predicted no building "
                f"pixels in this sector, so no damage score could be computed.")

    breakdown = (f"No Damage {_pct(z.share(1))}, Minor {_pct(z.share(2))}, "
                 f"Major {_pct(z.share(3))}, Destroyed {_pct(z.share(4))}")
    head = f"{z.name} scored {z.score:.2f} / 5"

    if z.priority == HIGH:
        reason = (f"{_pct(z.severe_share)} of its predicted building pixels were classified as "
                  f"Major Damage or Destroyed")
        rule = f"score ≥ {thresholds['high']:g}"
    elif z.priority == MEDIUM:
        reason = (f"{_pct(z.damaged_share)} of its predicted building pixels show some visible damage, "
                  f"with {_pct(z.severe_share)} classified as Major Damage or Destroyed")
        rule = f"{thresholds['medium']:g} ≤ score < {thresholds['high']:g}"
    else:
        reason = f"most of its predicted building pixels ({_pct(z.share(1))}) show no visible damage"
        rule = f"score < {thresholds['medium']:g}"

    note = ""
    if z.building_coverage < 0.01:
        note = (f" Note: buildings cover only {100 * z.building_coverage:.1f}% of this sector, "
                f"so the score rests on few pixels.")
    return f"{head} → {z.priority} priority ({rule}) because {reason}. Breakdown: {breakdown}.{note}"


def summarize(result: RankingResult) -> str:
    """Short analysis summary built from computed results only."""
    stats = overall_statistics(result.overall_counts)
    n_high, n_med, n_low, n_ins = (result.count(p) for p in PRIORITY_ORDER)
    scored = [z for z in result.zones if z.score is not None]

    if not scored:
        return ("The AI model did not predict any building pixels in this image, so no zone could be "
                "scored. Check that the image is a post-disaster satellite tile at a resolution similar to xBD.")

    def plural(n, word):
        return f"{n} {word}{'' if n == 1 else 's'}"

    parts = [
        f"The AI analysis identified {plural(n_high, 'high-priority zone')}, "
        f"{plural(n_med, 'medium-priority zone')} and {plural(n_low, 'low-priority zone')}"
        + (f"; {plural(n_ins, 'zone')} had insufficient data." if n_ins else ".")
    ]
    top = scored[0]
    if top.severe_share >= max(top.share(2), 1e-9):
        why = (f"due to the concentration of Major Damage and Destroyed predictions "
               f"({_pct(top.severe_share)} of its building pixels)")
    elif top.damaged_share > 0:
        why = f"with {_pct(top.damaged_share)} of its building pixels showing visible damage, mostly Minor"
    else:
        why = "although no damage was predicted there"
    parts.append(f"{top.name} received the highest severity score ({top.score:.2f}) {why}.")
    parts.append(
        f"Across the whole image, {_pct(stats['damaged_share'])} of predicted building pixels show visible "
        f"damage (Minor {_pct(stats['minor_share'])}, Major {_pct(stats['major_share'])}, "
        f"Destroyed {_pct(stats['destroyed_share'])})."
    )
    return " ".join(parts)


def zones_table(result: RankingResult) -> list[dict]:
    """Rows for the ranking table (plain dicts, easy to turn into a DataFrame)."""
    rows = []
    for z in result.zones:
        rows.append({
            "Rank": z.rank if z.rank is not None else "–",
            "Zone": z.name,
            "Score": round(z.score, 2) if z.score is not None else None,
            "Priority": z.priority,
            "Building px %": round(100 * z.building_coverage, 1),
            "No Damage %": round(100 * z.share(1), 1),
            "Minor %": round(100 * z.share(2), 1),
            "Major %": round(100 * z.share(3), 1),
            "Destroyed %": round(100 * z.share(4), 1),
        })
    return rows


__all__ = [
    "CLASS_NAMES", "DAMAGE_WEIGHTS", "DEFAULT_THRESHOLDS", "GRID_SIZE", "HIGH", "INSUFFICIENT", "LOW",
    "MEDIUM", "PRIORITY_COLORS", "PRIORITY_ORDER", "RankingResult", "Zone", "classify_priority",
    "count_classes", "damage_score", "explain_zone", "overall_statistics", "rank_zones",
    "split_into_zones", "summarize", "validate_thresholds", "zones_table",
]


# ---------------------------------------------------------------------------
# Plain-language text for the dashboard (same numbers, no technical terms).
# "Building area" = share of pixels the model classified as building; it is an
# area proportion, never a count of buildings.
# ---------------------------------------------------------------------------

def zone_insight(z: Zone) -> str:
    """One or two plain sentences explaining a zone's priority."""
    if z.score is None:
        return "No buildings were detected in this zone, so it was not scored."
    if z.priority == HIGH:
        text = (f"Severe damage pattern: {_pct(z.severe_share)} of the detected building area "
                f"shows major damage or destruction.")
    elif z.priority == MEDIUM:
        text = (f"Moderate damage: {_pct(z.damaged_share)} of the detected building area shows visible damage, "
                f"including {_pct(z.severe_share)} with major damage or destruction.")
    else:
        text = f"Mostly intact: {_pct(z.share(1))} of the detected building area shows no visible damage."
    if z.building_coverage < 0.01:
        text += " Very little building area was detected here, so this result is less reliable."
    return text


def plain_summary(result: RankingResult) -> str:
    """Short assessment summary in plain language, built from computed results only."""
    stats = overall_statistics(result.overall_counts)
    scored = [z for z in result.zones if z.score is not None]
    if not scored:
        return ("No buildings were detected in this image, so no zone could be assessed. "
                "Check that the image is a post-disaster satellite view of a built-up area.")

    n_zones = len(result.zones)
    n_high, n_med = result.count(HIGH), result.count(MEDIUM)
    n_ins = result.count(INSUFFICIENT)

    if n_high:
        first = f"{n_high} of {n_zones} zones {'is' if n_high == 1 else 'are'} high priority"
        first += f" and {n_med} {'is' if n_med == 1 else 'are'} medium priority." if n_med else "."
    elif n_med:
        first = f"No zone reached high priority; {n_med} of {n_zones} {'is' if n_med == 1 else 'are'} medium priority."
    else:
        first = "No zone reached medium or high priority; detected buildings appear largely intact."

    top = scored[0]
    if top.severe_share > 0:
        second = (f"{top.name} shows the most severe damage, with {_pct(top.severe_share)} of its detected "
                  f"building area showing major damage or destruction.")
    elif top.damaged_share > 0:
        second = (f"{top.name} shows the most damage, with {_pct(top.damaged_share)} of its detected "
                  f"building area showing minor damage.")
    else:
        second = ""

    third = (f"Across the whole image, {_pct(stats['damaged_share'])} of the detected building area shows "
             f"visible damage.")
    fourth = (f" {n_ins} {'zone contains' if n_ins == 1 else 'zones contain'} no detected buildings."
              if n_ins else "")
    return " ".join(p for p in (first, second, third) if p) + fourth
