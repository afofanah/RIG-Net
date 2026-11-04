"""geobase.py - schematic national outlines for the map figures.

Simplified admin-0 polygons (johan/world.geo.json, public domain), stored in data/geo/.
They are drawn only as a backdrop: every analysis uses the recorded seat coordinates.
"""
import json
from pathlib import Path
import numpy as np

GEO = Path(__file__).parent / "data" / "geo"
PANEL = {"WA2014": ("GIN", "LBR", "SLE"), "DRC1820": ("COD",)}


def rings(code):
    g = json.load(open(GEO / f"{code}.json"))["features"][0]["geometry"]
    polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    return [np.asarray(p[0], dtype=float) for p in polys]


def drc_zones(highlight=True):
    """Health-zone polygons for the 29 modelled DRC zones, plus their provincial context."""
    z = json.load(open(GEO / "DRC_zones.json"))["zones"]
    ctx = json.load(open(GEO / "DRC_provinces.json"))["rings"]
    return z, ctx


def wa_districts():
    """District polygons for the 61 modelled WA2014 regions, plus unmodelled context."""
    z = json.load(open(GEO / "WA_districts.json"))["zones"]
    ctx = json.load(open(GEO / "WA_context.json"))["rings"]
    return z, ctx


def draw(ax, panel, fill="#f2f5f7", edge="#9aa6b2", lw=0.8, zorder=0, alpha=1.0):
    """Outline the countries of a panel behind the data, leaving axis limits untouched."""
    keep = ax.has_data()
    xl, yl = ax.get_xlim(), ax.get_ylim()
    if panel == "WA2014":
        # real district boundaries for the three countries
        zz, ctx = wa_districts()
        for ring_set in ctx:
            for r in ring_set:
                a = np.asarray(r, float)
                ax.fill(a[:, 0], a[:, 1], facecolor=fill, edgecolor=edge,
                        lw=0.35, zorder=zorder, alpha=alpha)
        for ring_set in zz.values():
            for r in ring_set:
                a = np.asarray(r, float)
                ax.fill(a[:, 0], a[:, 1], facecolor="#e4edf2", edgecolor="#7f8fa0",
                        lw=0.5, zorder=zorder + 0.1, alpha=alpha)
        if keep: ax.set_xlim(*xl); ax.set_ylim(*yl)
        return
    if panel == "DRC1820":
        # real health-zone boundaries, with the three provinces as context
        z, ctx = drc_zones()
        for ring_set in ctx:
            for r in ring_set:
                a = np.asarray(r, float)
                ax.fill(a[:, 0], a[:, 1], facecolor=fill, edgecolor=edge,
                        lw=0.35, zorder=zorder, alpha=alpha)
        for ring_set in z.values():
            for r in ring_set:
                a = np.asarray(r, float)
                ax.fill(a[:, 0], a[:, 1], facecolor="#e4edf2", edgecolor="#7f8fa0",
                        lw=0.6, zorder=zorder + 0.1, alpha=alpha)
        if keep: ax.set_xlim(*xl); ax.set_ylim(*yl)
        return
    for code in PANEL.get(panel, ()):
        for r in rings(code):
            ax.fill(r[:, 0], r[:, 1], facecolor=fill, edgecolor=edge,
                    lw=lw, zorder=zorder, alpha=alpha, joinstyle="round")
    if keep:                       # never let the backdrop rescale the panel
        ax.set_xlim(*xl); ax.set_ylim(*yl)
