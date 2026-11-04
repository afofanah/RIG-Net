"""paper_analysis.py - comprehensive experimental analysis for the paper.

RQ1 accuracy vs 15 baselines        RQ5 spatio-temporal error structure
RQ2 contribution of each component  RQ6 interpretability / learned epidemiology
RQ3 epidemic phase and horizon      RQ7 hyper-parameter sensitivity
RQ4 validation vs test transfer     RQ8 cost (parameters, runtime)
"""
import json, sys
from pathlib import Path
from paths import FIG, PAPER
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from panels import label_axes
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).parent))
from baselines import BASELINES
from geobase import draw as draw_countries
from run_experiments import FOLDS, ABLATIONS, load, splits, train_eval, H_OUT, L_IN
from analyze import pooled, scores, dm_test, PHASE

ROOT = Path(__file__).parent; RES = ROOT / "results"
OUT = PAPER; FIG = OUT / "figs"; FIG.mkdir(parents=True, exist_ok=True)
BASE = list(BASELINES); REF = ["Naive", "MA3"]; SEEDS = [0, 1, 2, 3, 4]
OURS = "RIG-Net-GI"
plt.rcParams.update({"savefig.dpi": 400, "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2,
                     "axes.spines.top": False, "axes.spines.right": False})
C_OURS, C_BASE, C_REF = "#d9480f", "#1c7ed6", "#868e96"


def field(dn, model, seeds=SEEDS):
    """Per-(week, region) seed-averaged forecast and truth at each horizon, aligned on calendar weeks."""
    meta = json.load(open(ROOT / "data" / f"{dn}.json")); T, N = meta["T"], meta["N"]
    P = np.full((H_OUT, T, N), np.nan); Y = np.full((H_OUT, T, N), np.nan)
    for f in range(len(FOLDS[dn][0])):
        d = RES / "preds" / dn / f"fold{f}"
        files = [d / f"{model.replace(' ', '_').replace('/', '')}__s{s}.npy" for s in seeds]
        files = [x for x in files if x.exists()]
        if not files: return None, None, meta
        pr = np.mean([np.load(x) for x in files], 0); y = np.load(d / "Y.npy"); m = np.load(d / "M.npy"); O = np.load(d / "O.npy")
        for b, t0 in enumerate(O):
            for h in range(H_OUT):
                if m[b, h].any() and t0 + h < T:
                    P[h, t0 + h] = pr[b, h]; Y[h, t0 + h] = y[b, h]
    return P, Y, meta


def main():
    nums = {}

    # ---------------- RQ4: validation vs test transfer ----------------
    raw = pd.read_json(RES / "results.jsonl", lines=True)
    main_models = REF + BASE + [OURS, "RIG-Net", "RIG-Net-S (post-hoc)"]
    vt = raw[raw.model.isin(main_models)].groupby(["dataset", "model"]).agg(
        val=("val_MAE", "mean"), test=("MAE", "mean")).reset_index()
    nums["val_test_corr"] = {dn: float(g.val.corr(g.test, method="spearman")) for dn, g in vt.groupby("dataset")}
    nums["val_test_corr_pearson"] = {dn: float(g.val.corr(g.test)) for dn, g in vt.groupby("dataset")}
    vt.to_csv(RES / "val_vs_test.csv", index=False)

    with plt.rc_context({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 7.8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.2}):
     fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.95))
     for ax, dn in zip(axes, FOLDS):
        g = vt[vt.dataset == dn]
        ax.scatter(g.val, g.test, s=11, c=[C_OURS if m == OURS else (C_REF if m in REF else C_BASE) for m in g.model], zorder=3)
        for _, r in g.iterrows():
            if r.model in (OURS, "RIG-Net"):
                ax.annotate(r.model.replace("RIG-Net-GI", "ours"), (r.val, r.test), fontsize=5.8,
                            xytext=(2.5, 2), textcoords="offset points")
        lo, hi = min(g.val.min(), g.test.min()), max(g.val.max(), g.test.max())
        ax.plot([lo, hi], [lo, hi], color="k", lw=0.6, ls=":", zorder=1)
        ax.set_title(f"{dn} ($\\rho_s$={nums['val_test_corr'][dn]:.2f})", fontsize=8)
        ax.set_xlabel("validation MAE")
        ax.margins(0.12)
     axes[0].set_ylabel("test MAE")
     label_axes(fig, axes)
     fig.tight_layout(); fig.savefig(FIG / "val_vs_test.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)

    # ---------------- RQ5: spatio-temporal error structure ----------------
    for _k, dn in enumerate(FOLDS):
        Po, Yo, meta = field(dn, OURS); Pb, _, _ = field(dn, "MTGNN")
        weeks = pd.to_datetime(meta["weeks"]); nodes = meta["nodes"]
        h = 1                                                        # 2-week-ahead
        eo = np.abs(Po[h] - Yo[h]); eb = np.abs(Pb[h] - Yo[h])
        d = eb - eo                                                  # >0 : ours better
        order = np.argsort(-np.nansum(Yo[h], 0))[:min(18, len(nodes))]
        covered = ~np.all(np.isnan(d[:, order]), 1)
        M = d[covered][:, order].T; wk = weeks[covered]
        # (a) net gain per region, sorted; (b) weekly gain with its running total
        fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.9),
                                gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.36})
        net = np.nansum(d[:, order], 0)
        srt = np.argsort(net)
        lbl = [nodes[order[i]].split(":")[-1].replace("Western Area", "W. Area") for i in srt]
        axs[0].barh(range(len(srt)), net[srt],
                    color=["#1c7ed6" if v > 0 else "#e03131" for v in net[srt]],
                    edgecolor="black", linewidth=0.3)
        axs[0].axvline(0, color="k", lw=0.7)
        axs[0].set_yticks(range(len(srt))); axs[0].set_yticklabels(lbl, fontsize=7.4)
        axs[0].set_xlabel("cumulative error avoided (cases)", fontsize=9)
        axs[0].set_title(f"{dn}: net gain per region", fontsize=9.5)
        wk_gain = np.nansum(d[covered][:, order], 1)
        axs[1].bar(wk, wk_gain, width=pd.Timedelta(days=5),
                   color=["#1c7ed6" if v > 0 else "#e03131" for v in wk_gain], alpha=0.85)
        axs[1].axhline(0, color="k", lw=0.7)
        ax2 = axs[1].twinx()
        ax2.plot(wk, np.cumsum(np.nan_to_num(wk_gain)), color="#5f3dc4", lw=1.6)
        ax2.set_ylabel("running total", fontsize=9, color="#5f3dc4")
        ax2.tick_params(labelsize=8, colors="#5f3dc4")
        axs[1].set_ylabel("weekly gain (cases)", fontsize=9)
        axs[1].set_title(f"{dn}: when the gain accrues", fontsize=9.5)
        axs[1].tick_params(axis="x", rotation=30, labelsize=8)
        for a_ in axs: a_.tick_params(labelsize=8)
        label_axes(fig, axs, labels=[f"({chr(97 + 2 * _k + j)})" for j in range(2)])
        fig.savefig(FIG / f"sterr_{dn}.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)
        nums.setdefault("st", {})[dn] = {
            "region_weeks_scored": int(np.isfinite(d).sum()),
            "frac_region_weeks_better": float(np.nanmean(d > 0)),
            "frac_regions_better_on_average": float(np.mean(np.nanmean(d, 0) > 0)),
            "best_region": nodes[int(np.nanargmax(np.nansum(d, 0)))],
            "worst_region": nodes[int(np.nanargmin(np.nansum(d, 0)))]}

    # ---------------- RQ5b: error vs local incidence regime ----------------
    with plt.rc_context({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 7.8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.2}):
     fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.95))
     for ax, dn in zip(axes, FOLDS):
        Po, Yo, meta = field(dn, OURS); Pb, _, _ = field(dn, "MTGNN"); Pn, _, _ = field(dn, "Naive", [0])
        y = Yo.ravel(); ok = np.isfinite(y)
        bins = [(0, 0), (1, 2), (3, 9), (10, 29), (30, 10**6)]
        lab = ["0", "1-2", "3-9", "10-29", "$\\geq$30"]
        res = {}
        for name, Pm in (("RIG-Net-GI", Po), ("MTGNN", Pb), ("Naive", Pn)):
            e = np.abs(Pm - Yo).ravel()
            res[name] = [float(np.nanmean(e[ok & (y >= a) & (y <= b)])) for a, b in bins]
        x = np.arange(len(bins)); w = 0.26
        for k, (name, c) in enumerate([("RIG-Net-GI", C_OURS), ("MTGNN", C_BASE), ("Naive", C_REF)]):
            ax.bar(x + (k - 1) * w, res[name], w, color=c, label=name)
        ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=6.2); ax.set_yscale("log")
        ax.set_title(dn, fontsize=8); ax.set_xlabel("cases in region-week", fontsize=7.2)
        nums.setdefault("regime", {})[dn] = {"bins": lab, **res}
     axes[0].set_ylabel("MAE")
     axes[1].legend(frameon=False, fontsize=5.8, labelspacing=0.25, handlelength=1.0, borderpad=0.1)
     label_axes(fig, axes)
     fig.tight_layout(); fig.savefig(FIG / "regime.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)

    # ---------------- RQ6: learned R field + importation map (retrain seed 0) ----------------
    interp = {}; flowstore = {}
    for _k, dn in enumerate(FOLDS):
        z, meta = load(dn); ds = {k: z[k] for k in z.files}
        ds["U"] = np.load(ROOT / "data" / f"{dn}_interventions.npy"); gen = np.load(ROOT / "data" / f"{dn}_genomic.npz")
        T, N = ds["inc"].shape; Rf = np.full((T, N), np.nan); rho = np.zeros(N); A_acc = np.zeros((N, N)); nacc = 0
        betas, gates = [], []
        for f, s in enumerate(FOLDS[dn][0]):
            ds["G"] = gen[f"fold{f}"]
            for tv in (1, 4):
                p = ROOT / "data" / f"{dn}_genomic_tau{tv}.npz"
                if p.exists(): ds[f"G_tau{tv}"] = np.load(p)[f"fold{f}"]
            data = splits(ds["inc"], s, FOLDS[dn][1])
            _, _, w = train_eval(OURS, ds, data, 0, ABLATIONS[OURS])
            w.model.eval()
            with torch.no_grad(): o = w.model(w.to_x(data[2][0]), w.Uw["te"], return_diag=True)
            O = data[2][3]
            for b, t0 in enumerate(O):
                if t0 < T: Rf[t0] = o["R"][b, :, 0].numpy()
            rho += o["rho"].mean(0).numpy(); A_acc += o["A"].mean(0).numpy(); nacc += 1
            betas.append(w.model.beta_raw.detach().clamp_min(0).numpy()); gates.append(o["pi"].mean((0, 1)).numpy())
        rho /= nacc; A_acc /= nacc
        b = np.array(betas)
        interp[dn] = {"beta_mean": b.mean(0).tolist(), "beta_sd": b.std(0).tolist(),
                      "R_reduction_per_sd_pct": (100 * (1 - np.exp(-b.mean(0)))).tolist(),
                      "gate_geo_adp_dyn_gen": np.mean(gates, 0).tolist(),
                      "rho_mean": float(rho.mean()), "rho_min": float(rho.min()), "rho_max": float(rho.max())}
        # R field figure
        tot = np.nan_to_num(ds["inc"]).sum(0); order = np.argsort(-tot)[:min(14, N)]
        cov = ~np.all(np.isnan(Rf), 1)
        # (a) R trajectories for the largest regions; (b) share of regions above one, against incidence
        cov = ~np.all(np.isnan(Rf), 1)
        wk = pd.to_datetime(meta["weeks"])[cov]; Rc = Rf[cov]
        fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.9), gridspec_kw={"wspace": 0.30})
        pal = plt.cm.viridis(np.linspace(0.05, 0.85, min(6, len(order))))
        for c_, i in zip(pal, order[:6]):
            axs[0].plot(wk, Rc[:, i], lw=1.5, color=c_,
                        label=meta["nodes"][i].split(":")[-1].replace("Western Area", "W. Area"))
        axs[0].axhspan(1, 2.0, color="#e03131", alpha=0.055, lw=0, zorder=0)
        axs[0].axhline(1, color="k", lw=0.9, ls="--")
        axs[0].text(0.99, 0.515, "$R=1$: growth above, decline below", transform=axs[0].transAxes,
                    ha="right", va="bottom", fontsize=6.2, color="#495057",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
        axs[0].set_ylabel("learned $R_i(t)$", fontsize=9); axs[0].set_ylim(0, 2.0)
        axs[0].set_title(f"{dn}: $R$ in the largest regions", fontsize=9.5)
        axs[0].legend(frameon=False, fontsize=6.8, ncol=2)
        share = np.nanmean(Rc > 1, 1) * 100
        axs[1].fill_between(wk, 0, share, color="#e03131", alpha=0.30, lw=0)
        axs[1].plot(wk, share, color="#c92a2a", lw=1.7)
        axs[1].set_ylabel("regions with $R>1$ (\\%)", fontsize=9); axs[1].set_ylim(0, 100)
        ax2 = axs[1].twinx()
        ax2.plot(wk, np.nan_to_num(ds["inc"])[cov].sum(1), color="#495057", lw=1.2, ls=":")
        ax2.set_ylabel("observed weekly cases", fontsize=9, color="#495057")
        ax2.tick_params(labelsize=8, colors="#495057")
        below = np.where(share < 50)[0]
        if below.size and share[0] >= 50:
            k = int(below[0])
            axs[1].axvline(wk[k], color="#495057", lw=0.8, ls="--")
            axs[1].annotate(f"majority below $R=1$\n{wk[k]:%Y-%m-%d}", (wk[k], 92),
                            xytext=(6, 0), textcoords="offset points", fontsize=6.4,
                            color="#495057", va="top")
            nums.setdefault("Rshare", {}).setdefault(dn, {})["week_majority_below_1"] = str(wk[k].date())
        axs[1].annotate(f"{share[0]:.0f}%", (wk[0], share[0]), xytext=(3, -9),
                        textcoords="offset points", fontsize=6.4, color="#c92a2a")
        axs[1].annotate(f"{share[-1]:.0f}%", (wk[len(wk)-1], share[-1]), xytext=(-4, 7),
                        textcoords="offset points", fontsize=6.4, color="#c92a2a", ha="right")
        axs[1].set_title(f"{dn}: epidemic above threshold", fontsize=9.5)
        for a_ in axs: a_.tick_params(axis="x", rotation=30, labelsize=8); a_.tick_params(axis="y", labelsize=8)
        label_axes(fig, axs, labels=[f"({chr(97 + 2 * _k + j)})" for j in range(2)])
        fig.savefig(FIG / f"Rfield_{dn}.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)
        nums.setdefault("Rshare", {})[dn] = {
            "pct_weeks_majority_above_1": float(np.mean(share > 50) * 100),
            "share_first_week": float(share[0]), "share_last_week": float(share[-1])}
        # importation flow map -> stored, drawn as one stacked single-column figure below
        W = A_acc * rho[:, None]
        flowstore[dn] = dict(W=W, coords=ds["coords"], order=order, nodes=meta["nodes"], tot=tot, N=N)
        if dn == "WA2014":
            cc = np.array([n.split(":")[0] for n in meta["nodes"]])
            cross = float(W[cc[:, None] != cc[None, :]].sum() / W.sum())
            interp[dn]["cross_border_share_of_importation"] = cross

    with plt.rc_context({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                         "xtick.labelsize": 8.2, "ytick.labelsize": 8.2, "legend.fontsize": 7.6}):
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2), gridspec_kw={"wspace": 0.22})
        for ax, (dn, S) in zip(axes, flowstore.items()):
            W, coords, order, nodes, N = S["W"], S["coords"], S["order"], S["nodes"], S["N"]
            cc = np.array([n.split(":")[0] if ":" in n else "COD" for n in nodes])
            thr = np.sort(W.ravel())[-min(28, W.size)]
            wmax = W.max()
            inflow = W.sum(1)                                   # imported pressure received
            drawn = [(i, j) for i in range(N) for j in range(N) if i != j and W[i, j] >= thr]
            for i, j in sorted(drawn, key=lambda t: W[t]):       # strongest on top
                xborder = cc[i] != cc[j]
                ax.annotate("", xy=(coords[i, 1], coords[i, 0]),
                            xytext=(coords[j, 1], coords[j, 0]),
                            arrowprops=dict(arrowstyle="-|>", mutation_scale=3.4 + 2.4 * W[i, j] / wmax,
                                            lw=0.3 + 1.5 * W[i, j] / wmax,
                                            color="#c92a2a" if xborder else C_OURS,
                                            alpha=0.85 if xborder else 0.65,
                                            shrinkA=3, shrinkB=3), zorder=2)
            top5 = list(order[:5]); rest = [i for i in range(N) if i not in top5]
            ax.scatter(coords[rest, 1], coords[rest, 0],
                       s=8 + 120 * inflow[rest] / max(inflow.max(), 1e-9),
                       c="#0b7285", edgecolors="white", linewidths=0.3, zorder=3,
                       label="region ($\\propto$ imported)")
            # the five highest-incidence regions, numbered with a corner key
            import matplotlib.patheffects as pe
            from matplotlib.lines import Line2D as _L2D
            keys = []
            for q, i in enumerate(top5, start=1):
                ax.scatter([coords[i, 1]], [coords[i, 0]],
                           s=8 + 120 * inflow[i] / max(inflow.max(), 1e-9),
                           c="#0b7285", edgecolors="#f59f00", linewidths=1.3, zorder=4)
                ax.annotate(str(q), (coords[i, 1], coords[i, 0]), fontsize=5.6, ha="center",
                            va="center", color="white", zorder=5, fontweight="bold",
                            path_effects=[pe.withStroke(linewidth=1.1, foreground="#10243f")])
                keys.append(_L2D([], [], ls="none", marker="$%d$" % q, color="#495057", ms=4.6,
                                 label=nodes[i].split(":")[-1].replace("Western Area", "W. Area")))
            ax.add_artist(ax.legend(handles=keys, frameon=True, framealpha=0.9,
                                    edgecolor="#adb5bd", fontsize=5.6,
                                    loc="lower right" if dn != "WA2014" else "lower left",
                                    handletextpad=0.1, labelspacing=0.2, borderpad=0.3))
            ax.margins(0.14); draw_countries(ax, dn)
            ax.set_ylabel("latitude" if ax is axes[0] else "")
            ax.set_xlabel("longitude")
            ax.set_title(f"{dn}: importation flows $\\rho_iA_{{ij}}$", pad=4)
            from matplotlib.lines import Line2D
            h, l = ax.get_legend_handles_labels()
            h += [Line2D([], [], color=C_OURS, lw=2.0, label="within country"),
                  Line2D([], [], color="#c92a2a", lw=2.0, label="across a border")]
            l += ["within country", "across a border"]
            if dn != "WA2014":
                h, l = h[:-1], l[:-1]                           # DRC1820 is a single country
            ax.legend(h, l, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.10), ncol=3,
                      handletextpad=0.25, columnspacing=0.55, borderpad=0.1, fontsize=6.4)
        label_axes(fig, axes, inline=True)
        fig.savefig(FIG / "flows.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)

    nums["interp"] = interp

    # ---------------- RQ7: sensitivity ----------------
    sens = raw[raw.model.str.contains("=")]
    if len(sens):
        piv = sens.groupby(["dataset", "model"]).agg(test=("MAE", "mean"), val=("val_MAE", "mean")).reset_index()
        base = raw[raw.model == OURS].groupby("dataset").agg(test=("MAE", "mean"), val=("val_MAE", "mean"))
        rows = []
        for _, r in piv.iterrows():
            b = base.loc[r.dataset]
            rows.append({"dataset": r.dataset, "variant": r.model.replace("RIG-Net-GI ", ""),
                         "test_MAE": r.test, "delta_vs_default_pct": 100 * (r.test - b.test) / b.test,
                         "val_MAE": r.val})
        sdf = pd.DataFrame(rows); sdf.to_csv(RES / "sensitivity.csv", index=False)
        nums["sensitivity"] = sdf.to_dict("records")

    # ---------------- RQ8: cost ----------------
    cost = raw[raw.model.isin(BASE + [OURS])].groupby("model").agg(
        params=("n_params", "first"), sec=("train_sec", "mean")).reset_index()
    nums["cost"] = cost.to_dict("records")

    json.dump(nums, open(OUT / "paper_numbers_ext.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in nums.items() if k not in ("cost", "sensitivity")}, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
