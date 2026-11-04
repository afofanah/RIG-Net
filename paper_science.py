"""paper_science.py - spatial panels, genomic validation, theory validation,
time-step prediction and probabilistic metrics."""
import json, sys
from pathlib import Path
from paths import FIG, PAPER
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from panels import label_axes
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import nbinom, spearmanr, pearsonr
from scipy.interpolate import griddata

sys.path.insert(0, str(Path(__file__).parent))
from baselines import BASELINES
from run_experiments import FOLDS, ABLATIONS, load, splits, train_eval, H_OUT
from analyze import pooled, PHASE
from paper_analysis import field


ROOT = Path(__file__).parent; RES = ROOT / "results"
OUT = PAPER; FIG = OUT / "figs"; FIG.mkdir(parents=True, exist_ok=True)
BASE = list(BASELINES); SEEDS = [0, 1, 2, 3, 4]; OURS = "RIG-Net-GI"
plt.rcParams.update({"savefig.dpi": 400, "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2,
                     "axes.spines.top": False, "axes.spines.right": False})


def grid(coords, vals, n=220):
    ok = np.isfinite(vals)
    lo = coords.min(0) - 0.25; hi = coords.max(0) + 0.25
    gy, gx = np.mgrid[lo[0]:hi[0]:complex(0, n), lo[1]:hi[1]:complex(0, n)]
    z = griddata(coords[ok][:, ::-1], vals[ok], (gx, gy), method="linear")
    return gx, gy, z


def panel_map(ax, coords, vals, title, cmap="viridis", cb_label="", pts=True, norm=None):
    gx, gy, z = grid(coords, vals)
    im = ax.pcolormesh(gx, gy, z, cmap=cmap, shading="auto", norm=norm)
    if pts: ax.scatter(coords[:, 1], coords[:, 0], s=3.5, c="k", lw=0, alpha=0.65, zorder=3)
    ax.set_title(title, fontweight="bold"); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    cb = plt.colorbar(im, ax=ax, pad=0.015); cb.set_label(cb_label, fontsize=9.5); cb.ax.tick_params(labelsize=9)
    return im


def main():
    out = {}

    # =============== retrain once per fold, keep mu/alpha/R/A/gates ===============
    store = {}
    for dn in FOLDS:
        z, meta = load(dn); ds = {k: z[k] for k in z.files}
        ds["U"] = np.load(ROOT / "data" / f"{dn}_interventions.npy"); gen = np.load(ROOT / "data" / f"{dn}_genomic.npz")
        T, N = ds["inc"].shape
        MU = np.full((H_OUT, T, N), np.nan); YT = np.full((H_OUT, T, N), np.nan)
        ALPHA = np.zeros(N); Rf = np.full((T, N), np.nan); A_acc = np.zeros((N, N)); rho = np.zeros(N); k = 0
        gate_gen = np.full((len(FOLDS[dn][0]), N), np.nan)
        for f, s in enumerate(FOLDS[dn][0]):
            ds["G"] = gen[f"fold{f}"]
            for tv in (1, 4):
                p = ROOT / "data" / f"{dn}_genomic_tau{tv}.npz"
                if p.exists(): ds[f"G_tau{tv}"] = np.load(p)[f"fold{f}"]
            data = splits(ds["inc"], s, FOLDS[dn][1])
            _, _, w = train_eval(OURS, ds, data, 0, ABLATIONS[OURS])
            w.model.eval()
            Xte, Yte, Mte, O = data[2]
            with torch.no_grad(): o = w.model(w.to_x(Xte), w.Uw["te"], return_diag=True)
            mu = o["mu"].numpy(); al = o["phi"].numpy()
            ALPHA = al; A_acc += o["A"].mean(0).numpy(); rho += o["rho"].mean(0).numpy(); k += 1
            gate_gen[f] = o["pi"].mean(0)[:, 3].numpy()
            for b, t0 in enumerate(O):
                if t0 < T: Rf[t0] = o["R"][b, :, 0].numpy()
                for h in range(H_OUT):
                    if Mte[b, h].any() and t0 + h < T:
                        MU[h, t0 + h] = mu[b, h]; YT[h, t0 + h] = Yte[b, h].numpy()
        store[dn] = dict(meta=meta, inc=ds["inc"], coords=ds["coords"], MU=MU, YT=YT, ALPHA=ALPHA,
                         R=Rf, A=A_acc / k, rho=rho / k, gate_gen=np.nanmean(gate_gen, 0), U=ds["U"])

    # =============== (1) SPATIAL PANELS ===============
    for dn, S in store.items():
        coords = S["coords"]; meta = S["meta"]; inc = S["inc"]; N = inc.shape[1]
        tot = inc.sum(0)
        gsum = np.load(ROOT / "data" / f"{dn}_genomic.npz")[f"fold{len(FOLDS[dn][0])-1}"]
        gcov = (gsum.sum(1) > 0).astype(float)
        peak_wk = np.array([np.argmax(inc[:, i]) if tot[i] > 0 else np.nan for i in range(N)], float)
        err = np.nanmean(np.abs(S["MU"][1] - S["YT"][1]), 0)
        Pb, Yb, _ = field(dn, "MTGNN")
        gain = np.nanmean(np.abs(Pb[1] - Yb[1]), 0) - err
        Rmean = np.nanmean(S["R"], 0)
        fig, axes = plt.subplots(2, 3, figsize=(7.16, 5.4))
        panel_map(axes[0, 0], coords, np.log10(1 + tot), "(a) Cumulative incidence", "YlOrRd", "$\\log_{10}(1+$cases$)$")
        panel_map(axes[0, 1], coords, peak_wk, "(b) Week of local peak", "viridis", "weeks since start")
        deg = (gsum > 0).sum(1).astype(float)
        ax = axes[0, 2]
        ax.scatter(coords[gcov == 0, 1], coords[gcov == 0, 0], s=10, facecolors="none", edgecolors="#868e96", lw=0.6, label="not sequenced")
        sc = ax.scatter(coords[gcov == 1, 1], coords[gcov == 1, 0], s=6 + 7 * deg[gcov == 1], c=deg[gcov == 1], cmap="Purples",
                        edgecolors="#5f3dc4", lw=0.3, vmin=0)
        ax.set_title("(c) Genomic coverage", fontweight="bold"); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.legend(fontsize=8.5, frameon=False, loc="upper right", borderpad=0.2, handletextpad=0.4)
        cb = plt.colorbar(sc, ax=ax, pad=0.015); cb.set_label("genomically linked neighbours", fontsize=9.5); cb.ax.tick_params(labelsize=9)
        panel_map(axes[1, 0], coords, Rmean, "(d) Mean learned $R_i$", "RdYlBu_r", "$R_i$")
        panel_map(axes[1, 1], coords, S["rho"], "(e) Importation share $\\rho_i$", "BuPu", "$\\rho_i$")
        v = np.nanpercentile(np.abs(gain), 95) or 1
        panel_map(axes[1, 2], coords, gain, "(f) MAE gain over MTGNN", "RdBu", "cases (blue: ours better)",
                  norm=TwoSlopeNorm(0, -v, v))
        fig.suptitle({"WA2014": "Guinea / Liberia / Sierra Leone, 2014--15", "DRC1820": "North Kivu / Ituri / South Kivu, 2018--20"}[dn],
                     fontsize=10.5, fontweight="bold", y=1.0)
        label_axes(fig, axes)
        fig.tight_layout(); fig.savefig(FIG / f"spatial_{dn}.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)
        out.setdefault("spatial", {})[dn] = {
            "regions_sequenced": int(gcov.sum()), "N": int(N),
            "corr_R_with_incidence": float(spearmanr(Rmean, np.log1p(tot), nan_policy="omit").statistic),
            "corr_rho_with_gaincases": float(spearmanr(S["rho"], gain, nan_policy="omit").statistic)}

    # =============== (2) GENOMIC VALIDATION ===============
    # Does the genomic graph carry information the geographic kernel does not?
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))
    gstats = {}; COLD = {"WA2014": "#7048e8", "DRC1820": "#1c7ed6"}
    for dn in FOLDS:
        S = store[dn]; coords = S["coords"]; inc = S["inc"]; N = len(coords)
        G = np.load(ROOT / "data" / f"{dn}_genomic.npz")[f"fold{len(FOLDS[dn][0])-1}"]
        Wgeo = np.load(ROOT / "data" / f"{dn}.npz")["W"]
        D = np.zeros((N, N))
        for i in range(N):
            dy = np.radians(coords[:, 0] - coords[i, 0]); dx = np.radians(coords[:, 1] - coords[i, 1])
            D[i] = 6371 * 2 * np.arcsin(np.sqrt(np.sin(dy / 2) ** 2 + np.cos(np.radians(coords[i, 0])) *
                                                np.cos(np.radians(coords[:, 0])) * np.sin(dx / 2) ** 2))
        iu = np.triu_indices(N, 1)
        d = D[iu]; g = G[iu]; wg = Wgeo[iu]
        gen_e = g > 0; geo_e = wg > 0
        # (a) how far do the two graphs reach?
        for arr, ls, lab in ((d[geo_e], "--", "geographic edges"), (d[gen_e], "-", "genomic edges")):
            xs = np.sort(arr); axes[0].plot(xs, np.arange(1, len(xs) + 1) / len(xs), ls, lw=1.6,
                                            color=COLD[dn], label=f"{dn}: {lab}")
        # (b) distance decay of linkage strength, binned
        edges = np.array([0, 50, 100, 200, 400, 800]); cen = np.arange(len(edges) - 1)
        band = ["0--50", "50--100", "100--200", "200--400", "400--800"]
        mu = [np.nanmean(g[gen_e & (d >= lo) & (d < hi)]) if ((gen_e & (d >= lo) & (d < hi)).sum() > 2) else np.nan
              for lo, hi in zip(edges[:-1], edges[1:])]
        se = [np.nanstd(g[gen_e & (d >= lo) & (d < hi)]) / max(1, np.sqrt((gen_e & (d >= lo) & (d < hi)).sum()))
              for lo, hi in zip(edges[:-1], edges[1:])]
        axes[1].errorbar(cen + (0.06 if dn == "DRC1820" else -0.06), mu, yerr=se, marker="o", ms=4, lw=1.5,
                         capsize=2, color=COLD[dn], label=dn)
        # (c) complementarity
        novel = float((gen_e & ~geo_e).sum() / max(1, gen_e.sum()))
        far = float(np.mean(d[gen_e] > 200))
        gstats[dn] = {"genomic_edges": int(gen_e.sum()), "geographic_edges": int(geo_e.sum()),
                      "median_genomic_km": float(np.median(d[gen_e])), "median_geographic_km": float(np.median(d[geo_e])),
                      "frac_genomic_edges_not_in_geo_graph": novel, "frac_genomic_edges_beyond_200km": far,
                      "rho_linkage_distance": float(spearmanr(d[gen_e], g[gen_e]).statistic)}
    axes[0].set_xlabel("great-circle distance (km)"); axes[0].set_ylabel("cumulative fraction of edges")
    axes[0].set_title("(a) Reach of each graph", fontsize=9)
    axes[0].legend(frameon=False, fontsize=6.6, loc="lower right")
    axes[1].set_xticks(np.arange(len(band))); axes[1].set_xticklabels(band, rotation=30, ha="right", fontsize=7.6)
    axes[1].set_xlabel("distance band (km)"); axes[1].set_ylabel("mean linkage $G_{ij}$")
    axes[1].set_title("(b) Linkage decays weakly", fontsize=9); axes[1].legend(frameon=False)
    lab = list(gstats); x = np.arange(len(lab)); w_ = 0.36
    axes[2].bar(x - w_ / 2, [100 * gstats[k]["frac_genomic_edges_not_in_geo_graph"] for k in lab], w_,
                color="#7048e8", label="absent from geo. graph")
    axes[2].bar(x + w_ / 2, [100 * gstats[k]["frac_genomic_edges_beyond_200km"] for k in lab], w_,
                color="#f59f00", label="span $>$ 200 km")
    for i, k in enumerate(lab):
        axes[2].text(i - w_ / 2, 100 * gstats[k]["frac_genomic_edges_not_in_geo_graph"] + 1.5,
                     f"{100*gstats[k]['frac_genomic_edges_not_in_geo_graph']:.0f}", ha="center", fontsize=7.4)
        axes[2].text(i + w_ / 2, 100 * gstats[k]["frac_genomic_edges_beyond_200km"] + 1.5,
                     f"{100*gstats[k]['frac_genomic_edges_beyond_200km']:.0f}", ha="center", fontsize=7.4)
    axes[2].set_xticks(x); axes[2].set_xticklabels(lab); axes[2].set_ylabel("% of genomic edges")
    axes[2].set_ylim(0, 105); axes[2].set_title("(c) What the prior adds", fontsize=9)
    axes[2].legend(frameon=False, fontsize=7.2)
    label_axes(fig, axes)
    fig.tight_layout(); fig.savefig(FIG / "genomic_validation.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)
    out["genomic"] = gstats

    # =============== (3) THEORY VALIDATION ===============
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.3))
    th = {}
    w = np.array([0.185, 0.361, 0.258, 0.135, 0.061])

    def rho_C(Rbar):
        C = np.zeros((5, 5)); C[0] = Rbar * w; C[1:, :-1] = np.eye(4)
        return float(np.max(np.abs(np.linalg.eigvals(C))))

    # (a) the stability threshold, with the model's actual operating points
    rr = np.linspace(0.2, 2.5, 80)
    axes[0].plot(rr, [rho_C(x) for x in rr], color="#2f9e44", lw=1.6)
    axes[0].axhline(1, color="k", lw=0.6, ls=":"); axes[0].axvline(1, color="k", lw=0.6, ls=":")
    axes[0].fill_between(rr, 0, 1, where=np.array([rho_C(x) for x in rr]) < 1, color="#2f9e44", alpha=0.08)
    axt = axes[0].twinx()                      # the realised distribution, not just its median
    for dn, c in zip(FOLDS, ["#d9480f", "#1c7ed6"]):
        Rv = store[dn]["R"][np.isfinite(store[dn]["R"])].ravel()
        axt.hist(Rv, bins=np.linspace(0, 2.5, 50), color=c, alpha=0.30, lw=0,
                 density=True, label=f"{dn} ({Rv.size:,} region-weeks)")
        Rb = float(np.nanmedian(Rv))
        axes[0].plot([Rb], [rho_C(Rb)], marker="o", ms=6, color=c, zorder=5,
                     markeredgecolor="k", markeredgewidth=0.5)
        th[f"frac_R_below_1_{dn}"] = float((Rv < 1).mean())
    axt.set_yticks([]); axt.set_ylim(0, axt.get_ylim()[1] * 3.2)
    axt.legend(frameon=False, fontsize=5.4, loc="upper left", handlelength=1.0,
               handletextpad=0.4, borderpad=0.1)
    axes[0].set_xlabel("$\\bar R$"); axes[0].set_ylabel("$\\varrho(C)$")
    axes[0].set_title("(a) Thm. 1: stable iff $\\bar R<1$", fontsize=8)
    th["rho_at_R1"] = rho_C(1.0)

    # (b) measured error growth by regime, against the predicted envelopes
    for dn, c in zip(FOLDS, ["#d9480f", "#1c7ed6"]):
        S = store[dn]; Rb = np.nanmean(S["R"], 1)
        e = np.array([np.nanmean(np.abs(S["MU"][h] - S["YT"][h]), 1) for h in range(H_OUT)])
        for mask, ls, lab in ((Rb < 1.0, "-", "$\\bar R<1$"), (Rb >= 1.0, "--", "$\\bar R\\geq1$")):
            v = np.array([np.nanmean(e[h][mask]) for h in range(H_OUT)])
            if np.isfinite(v).all() and v[0] > 0:
                axes[1].plot(range(1, 5), v / v[0], ls, color=c, marker="o", ms=3, lw=1.4,
                             label=f"{'WA' if dn.startswith('WA') else 'DRC'}, {lab}")
    hs = np.arange(1, 5)
    axes[1].fill_between(hs, [rho_C(0.8) ** (h - 1) for h in hs], [rho_C(1.4) ** (h - 1) for h in hs],
                         color="k", alpha=0.08, label="Thm. 2 envelope")
    axes[1].set_xticks(hs); axes[1].set_xlabel("horizon $h$"); axes[1].set_ylabel("MAE$(h)$/MAE$(1)$")
    axes[1].set_title("(b) Thm. 2: error growth", fontsize=8); axes[1].legend(fontsize=5.6, frameon=False)

    # (c) APPLICATION: does the theory predict each block's error amplification?
    xs, ys, cs = [], [], []
    for dn, c in zip(FOLDS, ["#d9480f", "#1c7ed6"]):
        S = store[dn]
        for f, s0 in enumerate(FOLDS[dn][0]):
            sl = slice(s0, s0 + FOLDS[dn][1])
            for i in range(S["R"].shape[1]):                 # every region of every block
                Rb = np.nanmean(S["R"][sl, i])
                e1 = np.nanmean(np.abs(S["MU"][0][sl, i] - S["YT"][0][sl, i]))
                e4 = np.nanmean(np.abs(S["MU"][3][sl, i] - S["YT"][3][sl, i]))
                if np.isfinite(Rb) and np.isfinite(e1) and e1 > 0.5 and np.isfinite(e4):
                    xs.append(rho_C(Rb) ** 3); ys.append(e4 / e1); cs.append(c)
    xs, ys = np.array(xs), np.array(ys)
    axes[2].scatter(xs, ys, c=cs, s=9, alpha=0.45, lw=0, zorder=3)
    axes[2].set_yscale("log"); axes[2].set_xlim(0, np.percentile(xs, 99) * 1.05)
    lim = [max(1e-2, 0.8 * min(xs.min(), ys.min())), np.percentile(xs, 99) * 1.05]
    axes[2].plot(lim, lim, "k--", lw=0.9)
    rho_pred = float(spearmanr(xs, ys).statistic)
    axes[2].set_xlabel("predicted $\\varrho(C)^3$"); axes[2].set_ylabel("observed MAE$(4)$/MAE$(1)$")
    axes[2].set_title("(c) Bound is conservative", fontsize=8)
    th["block_amplification_spearman"] = rho_pred
    th["region_blocks_above_contraction_term"] = int((ys > xs).sum()); th["n_region_blocks"] = int(len(ys))

    # (d) Lemma 1 : fusion cannot destabilise message passing
    rng = np.random.default_rng(0); pts = []
    for dn in FOLDS:
        A = store[dn]["A"] if "A" in store[dn] else None
        A = A if A is not None else np.eye(5)
        N = A.shape[0]
        for _ in range(300):
            p = rng.dirichlet(np.ones(4)); q = rng.dirichlet(np.ones(4))
            mats = [A, np.eye(N)[rng.permutation(N)], np.full((N, N), 1 / N),
                    A.T / np.maximum(A.T.sum(1, keepdims=True), 1e-9)]
            M1 = sum(p[k] * mats[k] for k in range(4)); M2 = sum(q[k] * mats[k] for k in range(4))
            pts.append((np.abs(p - q).sum(), np.abs(M1 - M2).sum(1).max()))
    pts = np.array(pts)
    axes[3].scatter(pts[:, 0], pts[:, 1], s=3, c="#7048e8", alpha=0.3, lw=0)
    xx = np.linspace(0, 2, 20); axes[3].plot(xx, xx, "k--", lw=1)
    axes[3].fill_between(xx, xx, 2.1, color="#e03131", alpha=0.07)
    axes[3].text(1.0, 1.80, "bound violated\n(region empty)", fontsize=6.2, ha="center", color="#c92a2a")
    axes[3].set_xlabel("$\\|\\pi-\\pi'\\|_1$"); axes[3].set_ylabel("$\\|A-A'\\|_\\infty$")
    axes[3].set_title("(d) Lem. 1: fusion is safe", fontsize=8); axes[3].set_ylim(0, 2.1)
    th["lipschitz_violations"] = int((pts[:, 1] > pts[:, 0] + 1e-9).sum()); th["lipschitz_samples"] = len(pts)
    label_axes(fig, axes)
    fig.tight_layout(); fig.savefig(FIG / "theory.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)
    out["theory"] = th

    # =============== (4) TIME-STEP PREDICTION ===============
    for dn in FOLDS:
        S = store[dn]; meta = S["meta"]; inc = S["inc"]
        tot = inc.sum(0); top = np.argsort(-tot)[:4]
        weeks = pd.to_datetime(meta["weeks"])
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.8), sharex=True)
        for ax, j in zip(axes.ravel(), top):
            ax.plot(weeks, inc[:, j], color="k", lw=1.1, label="observed")
            for h, c in zip([0, 3], ["#d9480f", "#f59f00"]):
                mu = S["MU"][h][:, j]
                ax.plot(weeks, mu, color=c, lw=1.0, label=f"RIG-Net-GI $h{{=}}{h+1}$")
                lo = nbinom.ppf(0.05, 1 / S["ALPHA"][j], 1 / S["ALPHA"][j] / (1 / S["ALPHA"][j] + np.nan_to_num(mu)))
                hi = nbinom.ppf(0.95, 1 / S["ALPHA"][j], 1 / S["ALPHA"][j] / (1 / S["ALPHA"][j] + np.nan_to_num(mu)))
                if h == 0: ax.fill_between(weeks, lo, hi, color=c, alpha=0.16, lw=0, label="90% NB interval")
            Pb, _, _ = field(dn, "MTGNN")
            ax.plot(weeks, Pb[0][:, j], color="#1c7ed6", lw=0.8, ls="--", label="MTGNN $h{=}1$")
            for s in FOLDS[dn][0]:
                ax.axvspan(weeks[s], weeks[min(s + FOLDS[dn][1], len(weeks)) - 1], color="#fff4e6", zorder=0)
            ax.set_title(meta["nodes"][j].split(":")[-1], fontsize=10.5)
            ax.tick_params(axis="x", rotation=40, labelsize=9); ax.tick_params(axis="y", labelsize=9)
        axes[0, 0].legend(fontsize=8.5, frameon=False, ncol=2)
        axes[0, 0].set_ylabel("weekly cases", fontsize=10); axes[1, 0].set_ylabel("weekly cases", fontsize=10)
        axes[1, 0].set_xlabel("week", fontsize=10); axes[1, 1].set_xlabel("week", fontsize=10)
        fig.suptitle(f"{dn}: time-step predictions in the rolling test blocks (shaded)", fontsize=10.5, y=1.0)
        label_axes(fig, axes)
        fig.tight_layout(); fig.savefig(FIG / f"timestep_{dn}.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)

    # =============== (5) EXTRA METRICS ===============
    rows = []
    for dn in FOLDS:
        S = store[dn]; inc = S["inc"]
        scale = np.nanmean(np.abs(np.diff(inc, axis=0)))        # MASE denominator (seasonal-naive, lag 1)
        for model in [OURS, "MTGNN", "N-BEATS", "EpiGNN", "Naive"]:
            P, Y, _ = field(dn, model, [0] if model == "Naive" else SEEDS)
            e = np.abs(P - Y); ok = np.isfinite(e)
            mase = float(np.nanmean(e) / scale)
            den = (np.abs(P) + np.abs(Y)); sm = float(np.nanmean(np.where(den > 0, 2 * e / np.maximum(den, 1e-9), 0)) * 100)
            # peak-timing error per region (h=1)
            pk = []
            for j in range(inc.shape[1]):
                o = Y[0][:, j]; p = P[0][:, j]
                if np.nansum(o) > 20 and np.isfinite(p).sum() > 4:
                    pk.append(abs(np.nanargmax(p) - np.nanargmax(o)))
            rows.append({"dataset": dn, "model": model, "MASE": mase, "sMAPE": sm,
                         "peak_week_err": float(np.mean(pk)) if pk else np.nan})
        # probabilistic metrics for ours
        mu = S["MU"]; y = S["YT"]; al = S["ALPHA"][None, None, :]
        r = 1 / np.maximum(al, 1e-6); p = r / (r + np.nan_to_num(mu, nan=1.0))
        ok = np.isfinite(mu) & np.isfinite(y)
        lo = nbinom.ppf(0.05, r, p); hi = nbinom.ppf(0.95, r, p)
        cov = float(np.mean(((y >= lo) & (y <= hi))[ok]))
        qs = np.arange(0.05, 1.0, 0.05)
        wis = []
        for q in qs:
            qq = nbinom.ppf(q, r, p)
            wis.append(np.nanmean(2 * np.where(y[ok] >= qq[ok], q * (y[ok] - qq[ok]), (1 - q) * (qq[ok] - y[ok]))))
        out.setdefault("prob", {})[dn] = {"coverage90": cov, "WIS": float(np.mean(wis)),
                                          "mean_alpha": float(S["ALPHA"].mean())}
    dfm = pd.DataFrame(rows); dfm.to_csv(RES / "extra_metrics.csv", index=False)
    out["extra_metrics"] = dfm.to_dict("records")

    json.dump(out, open(OUT / "paper_numbers_sci.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
