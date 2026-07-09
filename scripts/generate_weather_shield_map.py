#!/usr/bin/env python3
"""Generate a geography-accurate Weather Shield Network map figure.

Downloads real prefecture boundaries from a public GeoJSON dataset (WGS84,
National Land Information Division of Japan, CC0) and caches locally so
subsequent runs work offline.  Pure stdlib + matplotlib + numpy — no geopandas.

Outputs (300 dpi, publication-ready):
  output/paper_fig6_weather_shield_map.png      ← English
  output/paper_fig6_weather_shield_map_ja.png   ← Japanese

Usage from repo root:
    python scripts/generate_weather_shield_map.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR   = REPO_ROOT / "output"
CACHE_DIR = OUT_DIR / ".cache"
OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ── GeoJSON source (all 47 JP prefectures, WGS84, CC0) ───────────────────────
GEOJSON_URL   = (
    "https://gist.githubusercontent.com/four4to6/"
    "d9bd390477e401bcb40fc5b0489fb52b/raw/"
    "00252f9582c7580504de4ea3e0d21141fd0af899/prefectures.geojson"
)
GEOJSON_CACHE = CACHE_DIR / "jp_prefectures.geojson"

# JIS prefecture codes
PREF_FUKUI    = 18
PREF_ISHIKAWA = 17
PREF_TOYAMA   = 16
PREF_GIFU     = 21
PREF_SHIGA    = 25
PREF_KYOTO    = 26
PREF_HYOGO    = 28   # fills Sea of Japan coast SW of Kyoto
PREF_AICHI    = 23   # fills SE corner gap south of Gifu
PREF_MIE      = 24   # fills remaining SE gap (Gifu/Aichi/Mie border)
PREF_NAGANO   = 20   # fills right-edge strip east of Gifu

# Bullet-train colour (RSI Maps style navy)
SHINKANSEN_BLUE = "#1B4F8A"


# ── Node definitions ──────────────────────────────────────────────────────────
NODES: dict[str, dict] = {
    "A": {
        "label_en": "Node A\nTojinbo (Coastal)",
        "label_ja": "ノードA\n東尋坊（沿岸）",
        "lon": 136.073, "lat": 36.174,
        "color": "#C0392B", "lost_k": 538.1,
    },
    "B": {
        "label_en": "Node B\nFukui Station (Hub)",
        "label_ja": "ノードB\n福井駅（拠点）",
        "lon": 136.219, "lat": 36.062,
        "color": "#2471A3", "lost_k": 313.7,
    },
    "C": {
        "label_en": "Node C\nKatsuyama (Mountain)",
        "label_ja": "ノードC\n勝山（山間）",
        "lon": 136.685, "lat": 36.017,
        "color": "#1E8449", "lost_k": 1.0,
    },
    "D": {
        "label_en": "Node D\nRainbow Line (Scenic)",
        "label_ja": "ノードD\nレインボーライン（景観）",
        "lon": 135.893, "lat": 35.540,
        "color": "#7D3C98", "lost_k": 13.1,
    },
}

# ── GeoJSON helpers ───────────────────────────────────────────────────────────

def _fetch_geojson() -> dict:
    """Return parsed GeoJSON, using local cache if present."""
    if GEOJSON_CACHE.exists():
        print(f"  Using cached GeoJSON: {GEOJSON_CACHE}")
        return json.loads(GEOJSON_CACHE.read_text(encoding="utf-8"))
    print("  Downloading prefecture boundaries …")
    with urllib.request.urlopen(GEOJSON_URL, timeout=30) as resp:
        raw = resp.read()
    GEOJSON_CACHE.write_bytes(raw)
    print(f"  Cached to {GEOJSON_CACHE}")
    return json.loads(raw.decode("utf-8"))


def _get_feature(geojson: dict, pref_code: int) -> dict | None:
    for feat in geojson["features"]:
        if feat["properties"].get("pref") == pref_code:
            return feat
    return None


def _exterior_rings(geometry: dict) -> list[list[tuple[float, float]]]:
    """Return list of exterior-ring coordinate lists from a GeoJSON geometry."""
    rings: list[list[tuple[float, float]]] = []
    gtype = geometry["type"]
    if gtype == "Polygon":
        rings.append(geometry["coordinates"][0])
    elif gtype == "MultiPolygon":
        for poly in geometry["coordinates"]:
            rings.append(poly[0])
    return rings


def _largest_ring(rings: list[list]) -> list[tuple[float, float]]:
    """Return the ring with the most vertices (the main land mass)."""
    return max(rings, key=len)


def _draw_pref(ax: plt.Axes, geometry: dict,
               facecolor: str, edgecolor: str,
               lw: float, alpha: float, zorder: int,
               main_only: bool = False) -> None:
    """Draw all exterior rings of a prefecture geometry as filled patches."""
    rings = _exterior_rings(geometry)
    if main_only:
        rings = [_largest_ring(rings)]
    for ring in rings:
        xy = [(lon, lat) for lon, lat in ring]
        patch = MplPolygon(xy, closed=True,
                           facecolor=facecolor, edgecolor=edgecolor,
                           linewidth=lw, alpha=alpha, zorder=zorder)
        ax.add_patch(patch)


# ── Bubble radius ─────────────────────────────────────────────────────────────

def _bubble_r(lost_k: float, max_k: float,
              r_min: float = 0.018, r_max: float = 0.063) -> float:
    return r_min + (r_max - r_min) * np.sqrt(max(lost_k, 0.5) / max_k)


# ── Arrow helpers ─────────────────────────────────────────────────────────────

def _curved_arrow(ax: plt.Axes, x0: float, y0: float,
                  x1: float, y1: float,
                  rad: float, color: str, lw: float, label: str = "") -> None:
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                connectionstyle=f"arc3,rad={rad}",
                                lw=lw, alpha=0.90),
                zorder=8)
    if label:
        mx = (x0 + x1) / 2 + (0.07 if rad > 0 else -0.07)
        my = (y0 + y1) / 2 + (0.05 if rad > 0 else -0.05)
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=7, color=color, style="italic", zorder=13,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.85))


# ── Main figure builder ───────────────────────────────────────────────────────

def _build_figure(geojson: dict, japanese: bool = False) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_facecolor("#D6EAF8")  # Sea of Japan

    # ── Neighboring prefectures (light backdrop, main polygon only) ────────
    # Each gets a distinct land colour so shared borders read clearly.
    # Adjacent pairs kept different: Ishikawa/Toyama, Gifu/Shiga, Shiga/Kyoto.
    backdrop = [
        (PREF_ISHIKAWA, "#D6C98A", "#B8AA6A", 0.85),   # golden wheat
        (PREF_TOYAMA,   "#A8C4B0", "#7AAA88", 0.80),   # sage green
        (PREF_GIFU,     "#C8A888", "#A07860", 0.78),   # warm terracotta
        (PREF_SHIGA,    "#B4C4A0", "#8AAA78", 0.75),   # olive (≠ Gifu, ≠ Kyoto)
        (PREF_KYOTO,    "#C4A8B4", "#A080A0", 0.75),   # soft mauve
        (PREF_HYOGO,    "#C0B8C8", "#9888A8", 0.65),   # muted lavender — fills SW coast gap
        (PREF_AICHI,    "#C8B8A0", "#A09078", 0.65),   # sandy tan — fills SE corner below Gifu
        (PREF_MIE,      "#C4B8A4", "#A09880", 0.62),   # similar tone — Gifu/Aichi/Mie border gap
        (PREF_NAGANO,   "#BEC8B0", "#9AAA88", 0.60),   # muted green — right-edge strip east of Gifu
    ]
    for code, fc, ec, alpha in backdrop:
        feat = _get_feature(geojson, code)
        if feat:
            # Gifu/Aichi are landlocked — draw all sub-polygons to close gaps
            _draw_pref(ax, feat["geometry"], facecolor=fc, edgecolor=ec,
                       lw=0.6, alpha=alpha, zorder=1,
                       main_only=(code not in (PREF_GIFU, PREF_AICHI, PREF_MIE)))

    # ── Fukui Prefecture (main subject) ────────────────────────────────────
    fukui_feat = _get_feature(geojson, PREF_FUKUI)
    if fukui_feat is None:
        raise RuntimeError("Fukui feature not found in GeoJSON (pref=18)")
    _draw_pref(ax, fukui_feat["geometry"],
               facecolor="#F0E8D0", edgecolor="#1A252F",
               lw=2.0, alpha=0.93, zorder=2)

    # ── Geographic text labels ─────────────────────────────────────────────
    # Water labels — italic, in actual sea/bay pixels
    for text, lon, lat in ([("Sea of Japan", 135.60, 35.95), ("Wakasa Bay", 135.82, 35.70)]
                           if not japanese else
                           [("日本海", 135.60, 35.95), ("若狭湾", 135.82, 35.70)]):
        ax.text(lon, lat, text, ha="center", va="center",
                fontsize=9.5, color="#1A5276", style="italic",
                alpha=0.85, zorder=12)

    # Prefecture name labels — all bold, semi-transparent matching-colour backgrounds.
    # Positions verified against GeoJSON boundaries visible in this viewport.
    # (text_en, text_ja, lon, lat, fs, color, bgcolor)
    pref_labels = [
        ("FUKUI",    "福井県",  136.42, 35.97, 11.0, "#2C3E50", "#F0E8D0"),   # upper-right body
        ("ISHIKAWA", "石川県",  136.62, 36.35,  9.0, "#6B5320", "#F5F0DC"),
        ("TOYAMA",   "富山県",  136.95, 36.68,  8.5, "#2A6B30", "#EBF5EB"),  # above Toyama station node
        ("GIFU",     "岐阜県",  137.02, 35.72,  8.5, "#7A3A10", "#F5EDE5"),
        ("SHIGA",    "滋賀県",  136.12, 35.30,  8.5, "#3A6A28", "#EBF0E5"),   # centred in olive region
        ("KYOTO",    "京都府",  135.55, 35.28,  8.5, "#6B2A6B", "#F0E8F0"),   # SW mauve area
    ]
    for en, ja, lon, lat, fs, col, bgcol in pref_labels:
        text = ja if japanese else en
        ax.text(lon, lat, text, ha="center", va="center",
                fontsize=fs, color=col, fontweight="bold",
                zorder=12,
                bbox=dict(boxstyle="round,pad=0.20", facecolor=bgcol,
                          edgecolor=col, linewidth=0.5, alpha=0.82))

    # ── Hub backbone spokes (dashed gray, node → Node B) ──────────────────
    hub = NODES["B"]
    max_k = max(nd["lost_k"] for nd in NODES.values())
    radii = {k: _bubble_r(nd["lost_k"], max_k) for k, nd in NODES.items()}
    r_hub = radii["B"]
    for key, nd in NODES.items():
        if key == "B":
            continue
        dx = hub["lon"] - nd["lon"]
        dy = hub["lat"] - nd["lat"]
        dist = np.hypot(dx, dy)
        # Stop the spoke at the circumference of B so the line is always visible
        end_lon = hub["lon"] - r_hub * dx / dist
        end_lat = hub["lat"] - r_hub * dy / dist
        ax.plot([nd["lon"], end_lon], [nd["lat"], end_lat],
                color="#BDC3C7", lw=1.2, alpha=0.6, linestyle="--",
                dashes=(4, 3), zorder=4)

    # ── Weather rerouting arrow: A → C ────────────────────────────────────
    _curved_arrow(ax,
                  NODES["A"]["lon"], NODES["A"]["lat"],
                  NODES["C"]["lon"], NODES["C"]["lat"],
                  rad=-0.20, color="#2471A3", lw=2.2)

    # ── Ishikawa spillover arrow ────────────────────────────────────────────
    # Start from central Ishikawa land mass; arc rightward (rad > 0 bows
    # southeast/inland) so the path stays over land, not the Sea of Japan.
    spill_src_lon, spill_src_lat = 136.42, 36.44
    ax.annotate("", xy=(NODES["A"]["lon"], NODES["A"]["lat"] + 0.025),
                xytext=(spill_src_lon, spill_src_lat),
                arrowprops=dict(arrowstyle="-|>", color="#E67E22", lw=2.0,
                                alpha=0.92,
                                connectionstyle="arc3,rad=0.30"),
                zorder=8)
    spill_txt = "石川スピルオーバー\nr = +0.549" if japanese else "Ishikawa Spillover\nr = +0.549"
    ax.text(spill_src_lon + 0.04, spill_src_lat + 0.02, spill_txt,
            ha="left", va="bottom", fontsize=8, color="#C0640A",
            fontweight="bold", zorder=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF5E7",
                      edgecolor="#E67E22", linewidth=1.2, alpha=0.93))

    # ── Node bubbles ───────────────────────────────────────────────────────
    # radii & max_k already computed above for spoke endpoints

    # (lon_delta, lat_delta, ha, va, add_r_lat)
    # add_r_lat=True → lbl_lat += r (for above-edge labels)
    lbl_props = {
        "A": (-0.10,  0.00, "right",  "center", False),  # left of bubble, shifted right
        "B": (-0.13, -0.10, "right",  "top",    True),   # lower-left corner, nudged left
        "C": ( 0.06,  0.00, "left",   "center", True),   # right, top edge
        "D": ( 0.10,  0.00, "left",   "center", False),  # right of bubble, nudged left
    }

    for key, nd in NODES.items():
        r = radii[key]
        circle = plt.Circle((nd["lon"], nd["lat"]), r,
                             facecolor=nd["color"], edgecolor="white",
                             linewidth=1.5, alpha=0.85, zorder=7)
        ax.add_patch(circle)

        # Node letter
        ax.text(nd["lon"], nd["lat"], key,
                ha="center", va="center",
                fontsize=10, fontweight="bold", color="white", zorder=12)

        # Label box
        off_lon, off_lat, ha, va, add_r = lbl_props[key]
        lbl = nd["label_ja"] if japanese else nd["label_en"]
        lbl_lon = nd["lon"] + off_lon
        lbl_lat = nd["lat"] + off_lat + (r if add_r else 0)
        ax.text(lbl_lon, lbl_lat, lbl,
                ha=ha, va=va,
                fontsize=8, fontweight="bold", color="#1C1C1C", zorder=13,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                          edgecolor=nd["color"], linewidth=0.9, alpha=0.92))

        # Lost-visitor sub-label (bold for legibility)
        if nd["lost_k"] >= 1.0:
            ax.text(nd["lon"], nd["lat"] - r - 0.010,
                    f"{nd['lost_k']:.0f}K lost/yr",
                    ha="center", va="top", fontsize=8,
                    fontweight="bold", color="#222", zorder=13,
                    bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                              edgecolor="none", alpha=0.80))

    # ── Hokuriku Shinkansen (opened Kanazawa→Tsuruga, March 2024) ─────────
    # GPS-approximate station positions (WGS84)
    shinkansen_stations = [
        ("Toyama",         "富山",        137.210, 36.700),   # Toyama (may clip to edge)
        ("Shin-Takaoka",   "新高岡",      136.993, 36.560),
        ("Kanazawa",       "金沢",        136.648, 36.579),
        ("Komatsu",        "小松",        136.451, 36.406),
        ("Kaga-Onsen",     "加賀温泉",    136.328, 36.312),
        ("Awara-Onsen",    "芦原温泉",    136.222, 36.215),
        ("Fukui",          "福井",        136.219, 36.062),   # = Node B
        ("Echizen-Takefu", "越前たけふ",  136.168, 35.897),
        ("Tsuruga",        "敦賀",        136.057, 35.645),
    ]
    sh_lons = [s[2] for s in shinkansen_stations]
    sh_lats = [s[3] for s in shinkansen_stations]

    # Outer white casing gives the characteristic double-rail look
    ax.plot(sh_lons, sh_lats, color="white",         lw=5.0, solid_capstyle="round",
            zorder=5)
    ax.plot(sh_lons, sh_lats, color=SHINKANSEN_BLUE, lw=2.8, solid_capstyle="round",
            zorder=5)

    # Station squares
    for _, _, slon, slat in shinkansen_stations:
        ax.plot(slon, slat, marker="s", markersize=4.5,
                color="white", markeredgecolor=SHINKANSEN_BLUE,
                markeredgewidth=1.4, zorder=6)

    # "→ Tokyo" label at the NE end
    to_tokyo = "→ 東京" if japanese else "→ Tokyo"
    ax.text(137.18, 36.69, to_tokyo, ha="left", va="center",
            fontsize=7, color=SHINKANSEN_BLUE, fontweight="bold", zorder=13,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor=SHINKANSEN_BLUE, linewidth=0.8, alpha=0.85))

    # Limited-express continuation Tsuruga → Kyoto/Osaka (JR Thunderbird route,
    # transfers at Tsuruga since Mar 2024; future Shinkansen extension planned).
    # Route goes SW from Tsuruga through Wakasa coastal corridor toward Kyoto.
    ltd_exp = [
        (136.057, 35.645),   # Tsuruga
        (135.930, 35.510),   # SW — Wakasa corridor
        (135.760, 35.340),   # deeper into Kyoto region
        (135.580, 35.210),   # extends to map edge — Kyoto/Osaka direction
    ]
    lx = [p[0] for p in ltd_exp]
    ly = [p[1] for p in ltd_exp]
    ax.plot(lx, ly, color="#5588CC", lw=1.8,
            solid_capstyle="round", zorder=5)

    # ── North compass arrow ────────────────────────────────────────────────
    # Positioned in the upper-left open sea (clear of all data/labels)
    cx, cy = 135.48, 36.38          # centre of compass in data coords
    arrow_len = 0.09
    ax.annotate("", xy=(cx, cy + arrow_len), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color="#333",
                                lw=1.8, mutation_scale=14),
                zorder=11)
    ax.text(cx, cy + arrow_len + 0.008, "N",
            ha="center", va="bottom", fontsize=9,
            fontweight="bold", color="#333", zorder=11)
    # Small circle base
    circle_c = plt.Circle((cx, cy), 0.022, facecolor="white",
                           edgecolor="#888", linewidth=0.8, zorder=10)
    ax.add_patch(circle_c)

    # ── Map extent ─────────────────────────────────────────────────────────
    # Extended east to show Toyama on the Shinkansen
    ax.set_xlim(135.40, 137.32)
    ax.set_ylim(35.20, 36.75)
    ax.set_aspect(1.0 / np.cos(np.radians(35.85)))

    # ── Axes cosmetics ─────────────────────────────────────────────────────
    ax.set_xlabel("経度 (°E)" if japanese else "Longitude (°E)",
                  fontsize=9, color="#555")
    ax.set_ylabel("緯度 (°N)" if japanese else "Latitude (°N)",
                  fontsize=9, color="#555")
    ax.tick_params(labelsize=8, color="#999", labelcolor="#666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#BBBBBB")

    # ── Inline line labels (no legend) ─────────────────────────────────────
    _lbl_style = dict(fontsize=7.5, fontweight="bold", zorder=13,
                      bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                edgecolor="none", alpha=0.88))

    # Weather Rerouting: centred on the A→C arc midpoint (arc bows south,
    # chord midpoint ≈ 136.38, 36.096; arc sits ~0.05° below that)
    ax.text(136.38, 36.12,
            "悪天候誘導" if japanese else "Weather Rerouting",
            ha="center", va="center", color="#2471A3", **_lbl_style)

    # Bullet Train: below the Toyama station square node, lowered
    ax.text(137.00, 36.50,
            "新幹線" if japanese else "Bullet Train",
            ha="center", va="center", color=SHINKANSEN_BLUE, **_lbl_style)

    # High Speed Train: lower and further right along the SW line
    ax.text(135.92, 35.36,
            "高速列車（敦賀→京阪）" if japanese else "High Speed Train",
            ha="right", va="center", color="#5588CC", **_lbl_style)

    # ── Title ──────────────────────────────────────────────────────────────
    if japanese:
        title = "4拠点天候シールド・ガバナンスアーキテクチャ\n需要サイド気象誘導ネットワーク，福井県"
    else:
        title = "Four-Node Weather Shield Governance Architecture\nDemand-Side Meteorological Rerouting Network, Fukui Prefecture, Japan"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)

    fig.tight_layout()
    return fig


def _apply_jp_font(fig: plt.Figure) -> None:
    import warnings

    import matplotlib.font_manager as fm
    candidates = ["Noto Sans CJK JP", "IPAexGothic", "IPAPGothic",
                  "Hiragino Sans", "Yu Gothic", "MS Gothic", "TakaoGothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((f for f in candidates if f in available), None)
    if chosen:
        plt.rcParams["font.family"] = chosen
    else:
        warnings.filterwarnings("ignore", "Glyph.*missing from font")


def main() -> None:
    print("=== Generating Weather Shield map figure ===")
    geojson = _fetch_geojson()

    out_en = OUT_DIR / "paper_fig6_weather_shield_map.png"
    out_ja = OUT_DIR / "paper_fig6_weather_shield_map_ja.png"

    fig_en = _build_figure(geojson, japanese=False)
    fig_en.savefig(out_en, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig_en)
    print(f"  Saved: {out_en}")

    _apply_jp_font(plt.figure())
    plt.close()
    fig_ja = _build_figure(geojson, japanese=True)
    _apply_jp_font(fig_ja)
    fig_ja.savefig(out_ja, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig_ja)
    print(f"  Saved: {out_ja}")

    print("Done.")


if __name__ == "__main__":
    main()
