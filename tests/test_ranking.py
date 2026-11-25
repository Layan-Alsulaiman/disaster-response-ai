"""Unit tests for ranking.py using small hand-made masks.

These masks exist only to check the arithmetic; they are never AI predictions.
"""

import numpy as np
import pytest

import ranking as rk


def mask_from_zones(zone_values, zone_px=4):
    """Build a 3x3-zone mask where each zone is filled per `zone_values[letter]`.

    zone_values maps "A".."I" -> list of class IDs filling that zone row-major
    (cycled to fill zone_px*zone_px pixels), or a single int.
    """
    m = np.zeros((3 * zone_px, 3 * zone_px), dtype=np.uint8)
    for i in range(9):
        letter = chr(ord("A") + i)
        r, c = divmod(i, 3)
        v = zone_values.get(letter, 0)
        vals = np.resize(np.array(v if isinstance(v, list) else [v], dtype=np.uint8), zone_px * zone_px)
        m[r * zone_px:(r + 1) * zone_px, c * zone_px:(c + 1) * zone_px] = vals.reshape(zone_px, zone_px)
    return m


# --- zone splitting ---------------------------------------------------------

def test_split_names_and_layout():
    zones = rk.split_into_zones(9, 9)
    assert [z[0] for z in zones] == [f"Zone {c}" for c in "ABCDEFGHI"]
    assert zones[0][3] == (0, 0, 3, 3)          # A top-left
    assert zones[4][3] == (3, 3, 6, 6)          # E centre
    assert zones[8][3] == (6, 6, 9, 9)          # I bottom-right
    assert (zones[3][1], zones[3][2]) == (1, 0)  # D is row 1, col 0


@pytest.mark.parametrize("h,w", [(10, 11), (1024, 1024), (7, 5), (1000, 1333)])
def test_split_covers_every_pixel_once(h, w):
    cover = np.zeros((h, w), dtype=int)
    for _, _, _, (x0, y0, x1, y1) in rk.split_into_zones(h, w):
        cover[y0:y1, x0:x1] += 1
    assert (cover == 1).all()


# --- scoring ---------------------------------------------------------------

def test_damage_score_formula():
    counts = {0: 100, 1: 10, 2: 5, 3: 3, 4: 2}           # building = 20
    expected = (1 * 5 + 3 * 3 + 5 * 2) / 20
    assert rk.damage_score(counts) == pytest.approx(expected)


def test_score_ignores_background_and_bounds():
    assert rk.damage_score({0: 999, 1: 1, 2: 0, 3: 0, 4: 0}) == 0.0
    assert rk.damage_score({0: 999, 1: 0, 2: 0, 3: 0, 4: 7}) == 5.0
    assert rk.damage_score({0: 50, 1: 0, 2: 0, 3: 0, 4: 0}) is None


@pytest.mark.parametrize("score,expected", [
    (5.0, rk.HIGH), (2.5, rk.HIGH), (2.4999, rk.MEDIUM), (1.0, rk.MEDIUM),
    (0.9999, rk.LOW), (0.0, rk.LOW), (None, rk.INSUFFICIENT),
])
def test_classify_priority_boundaries(score, expected):
    assert rk.classify_priority(score, rk.DEFAULT_THRESHOLDS) == expected


def test_custom_thresholds():
    assert rk.classify_priority(2.0, {"high": 1.5, "medium": 0.5}) == rk.HIGH
    assert rk.classify_priority(0.6, {"high": 1.5, "medium": 0.5}) == rk.MEDIUM


@pytest.mark.parametrize("bad", [{"high": 1.0, "medium": 1.0}, {"high": 6, "medium": 1},
                                 {"high": 2, "medium": -1}, {"high": 1, "medium": 2}])
def test_invalid_thresholds_rejected(bad):
    with pytest.raises(ValueError):
        rk.validate_thresholds(bad)


# --- full ranking ----------------------------------------------------------

def test_rank_zones_scores_priorities_and_order():
    m = mask_from_zones({
        "A": 4,               # all destroyed            -> 5.0 High
        "B": [1, 1, 1, 2],    # 3/4 no dmg, 1/4 minor    -> 0.25 Low
        "D": [3, 4],          # half major, half destroy -> 4.0 High
        "E": [1, 3],          # half none, half major    -> 1.5 Medium
        "F": [0, 0, 0, 1],    # sparse undamaged         -> 0.0 Low
        # C, G, H, I background only                    -> Insufficient Data
    })
    res = rk.rank_zones(m)
    by = {z.letter: z for z in res.zones}

    assert by["A"].score == pytest.approx(5.0) and by["A"].priority == rk.HIGH
    assert by["D"].score == pytest.approx(4.0) and by["D"].priority == rk.HIGH
    assert by["E"].score == pytest.approx(1.5) and by["E"].priority == rk.MEDIUM
    assert by["B"].score == pytest.approx(0.25) and by["B"].priority == rk.LOW
    assert by["F"].score == pytest.approx(0.0) and by["F"].priority == rk.LOW
    for letter in "CGHI":
        assert by[letter].score is None and by[letter].priority == rk.INSUFFICIENT
        assert by[letter].rank is None

    assert [z.letter for z in res.zones[:5]] == ["A", "D", "E", "B", "F"]
    assert [z.rank for z in res.zones[:5]] == [1, 2, 3, 4, 5]
    assert all(z.priority == rk.INSUFFICIENT for z in res.zones[5:])
    assert res.count(rk.HIGH) == 2 and res.count(rk.INSUFFICIENT) == 4


def test_counts_per_zone_sum_to_image():
    rng = np.random.default_rng(1)
    m = rng.integers(0, 5, (31, 47)).astype(np.uint8)
    res = rk.rank_zones(m)
    total = {c: sum(z.counts[c] for z in res.zones) for c in range(5)}
    assert total == rk.count_classes(m) == res.overall_counts
    assert sum(z.total_pixels for z in res.zones) == m.size


def test_tie_break_prefers_more_destroyed():
    # A: [4,3,2,2] -> (5+3+1+1)/4 = 2.5 with 25% destroyed
    # B: [4,1]     -> 5/2       = 2.5 with 50% destroyed  -> B must rank first despite name order
    m = mask_from_zones({"A": [4, 3, 2, 2], "B": [4, 1]})
    res = rk.rank_zones(m)
    a, b = res.by_name("A"), res.by_name("B")
    assert a.score == pytest.approx(b.score)
    assert b.share(4) > a.share(4)
    assert res.zones[0].letter == "B"


def test_thresholds_change_priority_not_scores():
    m = mask_from_zones({"A": [1, 3]})  # 1.5
    assert rk.rank_zones(m).by_name("A").priority == rk.MEDIUM
    res = rk.rank_zones(m, {"high": 1.2, "medium": 0.5})
    assert res.by_name("A").priority == rk.HIGH
    assert res.by_name("A").score == pytest.approx(1.5)


def test_rank_zones_rejects_bad_masks():
    with pytest.raises(ValueError):
        rk.rank_zones(np.zeros((3, 3, 1), dtype=np.uint8))
    with pytest.raises(ValueError):
        rk.rank_zones(np.full((3, 3), 7, dtype=np.uint8))


# --- statistics, explanations, summary --------------------------------------

def test_overall_statistics():
    counts = {0: 60, 1: 20, 2: 10, 3: 6, 4: 4}
    s = rk.overall_statistics(counts)
    assert s["building_pixels"] == 40
    assert s["building_coverage"] == pytest.approx(0.4)
    assert s["damaged_share"] == pytest.approx(0.5)
    assert s["minor_share"] == pytest.approx(0.25)
    assert s["major_share"] == pytest.approx(0.15)
    assert s["destroyed_share"] == pytest.approx(0.10)
    assert s["overall_score"] == pytest.approx((10 + 18 + 20) / 40)


def test_overall_statistics_no_buildings():
    s = rk.overall_statistics({0: 10, 1: 0, 2: 0, 3: 0, 4: 0})
    assert s["damaged_share"] == 0.0 and s["overall_score"] is None


def test_explanations_use_actual_numbers():
    m = mask_from_zones({"D": [3, 4], "E": [1, 3], "B": [1, 1, 1, 2]})
    res = rk.rank_zones(m)
    d = res.by_name("D").explanation
    assert "High" in d and "4.00" in d and "100%" in d and "Major Damage or Destroyed" in d
    assert "Medium" in res.by_name("E").explanation
    assert "75%" in res.by_name("B").explanation
    assert "Insufficient Data" in res.by_name("A").explanation


def test_summary_mentions_counts_and_top_zone():
    m = mask_from_zones({"A": 4, "D": [3, 4], "E": [1, 3], "B": [1, 1, 1, 2]})
    text = rk.summarize(rk.rank_zones(m))
    assert "2 high-priority zones" in text
    assert "1 medium-priority zone" in text
    assert "Zone A received the highest severity score (5.00)" in text
    assert "building pixels" in text and "buildings" not in text   # pixels, never building counts


def test_summary_without_buildings():
    text = rk.summarize(rk.rank_zones(np.zeros((9, 9), dtype=np.uint8)))
    assert "did not predict any building pixels" in text


def test_zones_table_rows():
    m = mask_from_zones({"A": 4})
    rows = rk.zones_table(rk.rank_zones(m))
    assert len(rows) == 9
    assert rows[0]["Zone"] == "Zone A" and rows[0]["Score"] == 5.0 and rows[0]["Destroyed %"] == 100.0
    assert rows[-1]["Score"] is None and rows[-1]["Rank"] == "–"
