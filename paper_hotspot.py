"""paper_hotspot.py - where the cases are, and where that concentration is statistically unusual.

Left panel per outbreak: region seats by country, with Voronoi cells approximating the
administrative zones and the study window as a convex hull.
Right panel: case-weighted kernel density, with Getis-Ord G_i^* hot and cold spots of
cumulative incidence (8 nearest neighbours, Benjamini-Hochberg correction at 5%).
"""
import json, sys
from pathlib import Path
from paths import FIG, PAPER
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from panels import label_axes
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import Voronoi, ConvexHull, cKDTree
from scipy.stats import norm
from matplotlib.path import Path as MplPath

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import load
from geobase import draw as draw_countries

OUT = PAPER; FIG = OUT / "figs"
CC = {"GIN": "Guinea", "LBR": "Liberia", "SLE": "Sierra Leone", "COD": "DR Congo"}
COL = {"GIN": "#1c7ed6", "LBR": "#2f9e44", "SLE": "#862e9c", "COD": "#d9480f"}
MK = {"GIN": "o", "LBR": "s", "SLE": "D", "COD": "o"}
plt.rcParams.update({"savefig.dpi": 400, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.6,
                     "axes.spines.top": False, "axes.spines.right": False})


def getis_ord(x, coords, k=8):
    """G_i^* on log1p counts with binary kNN weights (self included), BH-corrected at 5%."""
    n = len(x); tree = cKDTree(coords)
    _, nb = tree.query(coords, k=min(k + 1, n))
    xbar, s = x.mean(), x.std(ddof=0)
    z = np.zeros(n)
    for i in range(n):
        w = np.zeros(n); w[nb[i]] = 1.0
        W, W2 = w.sum(), (w ** 2).sum()
        denom = s * np.sqrt(max((n * W2 - W ** 2) / (n - 1), 1e-12))
        z[i] = ((w * x).sum() - xbar * W) / denom if denom > 0 else 0.0
    p = 2 * (1 - norm.cdf(np.abs(z)))
    order = np.argsort(p); m = len(p); bh = np.empty(m)
    crit = np.arange(1, m + 1) / m * 0.05
    passed = p[order] <= crit
    cutoff = np.max(np.where(passed)[0]) if passed.any() else -1
    sig = np.zeros(m, bool)
    if cutoff >= 0: sig[order[:cutoff + 1]] = True
    return z, p, sig


def voronoi_cells(ax, coords, hull_path, lo, hi):
    span = 0.45 * np.hypot(*(hi - lo))
    pad = np.array([[lo[0] - 10, lo[1] - 10], [lo[0] - 10, hi[1] + 10],
                    [hi[0] + 10, lo[1] - 10], [hi[0] + 10, hi[1] + 10]])
    vor = Voronoi(np.vstack([coords[:, ::-1], pad]))
    for ridge in vor.ridge_vertices:
        if -1 in ridge: continue
        seg = vor.vertices[ridge]
        if all(hull_path.contains_point(p) for p in seg) and np.hypot(*(seg[1] - seg[0])) < span:
            ax.plot(seg[:, 0], seg[:, 1], color="#adb5bd", lw=0.6, zorder=1)


def main():
    out = {}
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 6.6),
                             gridspec_kw={"hspace": 0.34, "wspace": 0.26})
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as pe
    for row, dn in enumerate(("WA2014", "DRC1820")):
        z, meta = load(dn)
        inc = z["inc"]; coords = z["coords"]
        nm = [n.split(":")[-1].replace("Western Area", "W. Area") for n in meta["nodes"]]
        tot = inc.sum(0)
        pts_true = coords[:, ::-1].astype(float)           # (lon, lat) as recorded
        pts = pts_true.copy()
        # relax co-located seats for DISPLAY ONLY; the statistic below uses pts_true
        span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1e-6); rmin = 0.075 * span
        for _ in range(260):
            dvec = pts[:, None, :] - pts[None, :, :]
            dist = np.hypot(dvec[..., 0], dvec[..., 1]); np.fill_diagonal(dist, np.inf)
            ii, jj = np.where(dist < rmin)
            if len(ii) == 0: break
            push = np.zeros_like(pts)
            for a_, b_ in zip(ii, jj):
                v = pts[a_] - pts[b_]; n = np.hypot(*v)
                v = v / n if n > 1e-9 else np.random.default_rng(a_ + b_).normal(size=2)
                push[a_] += 0.5 * (rmin - n) * v
            pts = pts + 0.35 * push
        hull = ConvexHull(pts); hp = MplPath(pts[hull.vertices])
        lo, hi = pts.min(0), pts.max(0)
        pad = 0.10 * np.maximum(hi - lo, 1e-6)
        xlim, ylim = (lo[0]-pad[0], hi[0]+pad[0]), (lo[1]-pad[1], hi[1]+pad[1])
        hv = np.append(hull.vertices, hull.vertices[0])

        zsc, p, sig = getis_ord(np.log1p(tot), pts_true, k=8)   # true coordinates
        hot = sig & (zsc > 0); cold = sig & (zsc < 0)

        def key(ax, idx, colour):
            """number the listed regions on the map and name them in a corner key"""
            h = []
            for q, i in enumerate(idx, start=1):
                ax.scatter([pts[i, 0]], [pts[i, 1]], s=165, facecolors="none",
                           edgecolors=colour, linewidths=1.5, zorder=6)
                ax.annotate(str(q), (pts[i, 0], pts[i, 1]), fontsize=7.4, fontweight="bold",
                            color="#10243f", ha="center", va="center", zorder=7,
                            path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
                h.append(Line2D([], [], marker="o", ls="none", ms=7, markerfacecolor="none",
                                markeredgecolor=colour, markeredgewidth=1.5,
                                label=f"{q}.\u2009{nm[i]}"))
            ax.legend(handles=h, frameon=True, framealpha=0.92, edgecolor="#adb5bd",
                      fontsize=7.0, loc="upper left", handletextpad=0.4,
                      borderpad=0.35, labelspacing=0.28).set_zorder(8)

        # ---------- left: cumulative incidence over the region graph ----------
        ax = axes[row, 0]
        draw_countries(ax, dn)
        sc = ax.scatter(pts[:, 0], pts[:, 1], s=22 + 230 * tot / max(tot.max(), 1),
                        c=np.log10(1 + tot), cmap="YlOrRd", edgecolors="black",
                        linewidths=0.45, zorder=4)
        cb = plt.colorbar(sc, ax=ax, pad=0.015, fraction=0.045)
        cb.set_label("cases (log$_{10}$)", fontsize=7.6, labelpad=1); cb.ax.tick_params(labelsize=7)
        key(ax, list(np.argsort(-tot)[:4]), "#1864ab")
        ax.set_title(f"{dn}: cumulative incidence", fontsize=9.5)
        ax.set_ylabel("Latitude", fontsize=9)

        # ---------- right: Getis-Ord statistic, same axes ----------
        ax = axes[row, 1]
        draw_countries(ax, dn)
        vmax = max(np.abs(zsc).max(), 1e-6)
        sc2 = ax.scatter(pts[:, 0], pts[:, 1], s=46, c=zsc, cmap="RdBu_r",
                         norm=TwoSlopeNorm(0, -vmax, vmax), edgecolors="#adb5bd",
                         linewidths=0.4, zorder=4)
        ax.scatter(pts[sig, 0], pts[sig, 1], s=46, facecolors="none", edgecolors="black",
                   linewidths=1.3, zorder=5)
        cb = plt.colorbar(sc2, ax=ax, pad=0.015, fraction=0.045)
        cb.set_label("$G_i^{\\star}$ $z$-score", fontsize=7.6, labelpad=1); cb.ax.tick_params(labelsize=7)
        key(ax, list(np.where(hot)[0][np.argsort(-tot[hot])][:4]), "#c92a2a")
        ax.set_title(f"{dn}: {int(hot.sum())} hot, {int(cold.sum())} cold "
                     f"(of {len(tot)})", fontsize=9.5)

        for ax in axes[row]:
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.set_xlabel("Longitude", fontsize=9); ax.tick_params(labelsize=8)
        out[dn] = {"n_regions": int(len(tot)), "hot": [nm[i] for i in np.where(hot)[0]],
                   "cold_n": int(cold.sum()), "not_sig_n": int((~sig).sum()),
                   "hot_case_share": float(tot[hot].sum() / max(tot.sum(), 1))}
    handles = [Line2D([], [], marker="o", ls="none", ms=7, markerfacecolor="#fdbf6f",
                      markeredgecolor="k", label="region (area $\\propto$ cases)"),
               Line2D([], [], color="#7f8fa0", lw=1.2, label="administrative boundary"),
               Line2D([], [], marker="o", ls="none", ms=7, markerfacecolor="none",
                      markeredgecolor="k", markeredgewidth=1.3,
                      label="significant at 5% (BH-corrected)")]
    fig.subplots_adjust(hspace=0.68, top=0.90)   # row gap, plus room for the legend above
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 0.985), fontsize=8.2)
    label_axes(fig, axes, relayout=False)
    fig.savefig(FIG / "hotspot.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)
    json.dump(out, open(OUT / "paper_numbers_hotspot.json", "w"), indent=1)
    print(json.dumps(out, indent=1)[:600])


if __name__ == "__main__":
    main()
