"""paper_validity.py - the checks a reviewer will ask for.

(1) leakage timeline per fold, with the genomic cut-off marked
(2) learned R validated against an EpiEstim-style estimator
(3) calibration: reliability curve + PIT histogram
(4) critical-difference diagram (Friedman + Nemenyi) over folds x datasets
(5) seed/fold variability
(6) genomic dose-response and graph-construction sensitivity
(7) accuracy-cost Pareto front
(8) Guinea 2021 zero-shot transfer
"""
import json, sys
from pathlib import Path
from paths import FIG, PAPER
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from panels import label_axes
from scipy.stats import nbinom, rankdata, friedmanchisquare, spearmanr, pearsonr
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).parent))
from baselines import BASELINES
from run_experiments import FOLDS, ABLATIONS, load, splits, train_eval, H_OUT, L_IN, ROOT
from analyze import pooled

OUT = PAPER; FIG = OUT / "figs"; RES = ROOT / "results"
BASE = list(BASELINES); SEEDS = [0, 1, 2, 3, 4]; OURS = "RIG-Net-GI"
W_GI = np.array([0.185, 0.361, 0.258, 0.135, 0.061])
plt.rcParams.update({"savefig.dpi": 400, "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
                     "axes.spines.top": False, "axes.spines.right": False})


def epiestim(y, tau=3, a=1.0, b=5.0):
    """Cori et al. (2013) posterior-mean R with a sliding window tau and Gamma(a,b) prior."""
    T = len(y); R = np.full(T, np.nan)
    lam = np.array([sum(W_GI[s] * y[t - s - 1] for s in range(min(5, t))) for t in range(T)])
    for t in range(T):
        lo = max(0, t - tau + 1)
        I = y[lo:t + 1].sum(); L = lam[lo:t + 1].sum()
        if L > 1e-6: R[t] = (a + I) / (1 / b + L)
    return R


def main():
    out = {}

    # ---------------- (1) leakage timeline ----------------
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5), gridspec_kw={"width_ratios": [37, 99]})
    gsum = {dn: json.load(open(ROOT / "data" / f"{dn}_genomic_summary.json")) for dn in FOLDS}
    for ax, dn in zip(axes, FOLDS):
        meta = json.load(open(ROOT / "data" / f"{dn}.json")); weeks = pd.to_datetime(meta["weeks"])
        inc = np.load(ROOT / "data" / f"{dn}.npz")["inc"]
        starts, blk = FOLDS[dn]
        for f, s in enumerate(starts):
            v0 = splits(inc, s, blk)[3][0]
            y = len(starts) - f
            ax.barh(y, v0 - L_IN, left=L_IN, height=0.55, color="#adb5bd", label="train" if f == 0 else None)
            ax.barh(y, s - v0, left=v0, height=0.55, color="#4c6ef5", label="validation" if f == 0 else None)
            ax.barh(y, min(blk, len(weeks) - s), left=s, height=0.55, color="#d9480f", label="test" if f == 0 else None)
            cut = list(gsum[dn]["folds"].keys())[f]
            ax.plot([v0, v0], [y - 0.45, y + 0.45], color="#7048e8", lw=2)
            ax.text(v0 - 0.4, y, f"{gsum[dn]['folds'][cut]['genomes_before_cutoff']}",
                    fontsize=5.8, ha="right", va="center", color="#7048e8")
        ax.set_yticks(range(1, len(starts) + 1)); ax.set_yticklabels([f"fold {len(starts)-i}" for i in range(len(starts))])
        step = max(1, len(weeks) // 8)
        ax.set_xticks(range(0, len(weeks), step)); ax.set_xticklabels([str(w.date()) for w in weeks[::step]], rotation=35, fontsize=6)
        ax.set_title(dn, fontsize=8.5, pad=6)
        ax.set_xlim(0, len(weeks))
    for ax, dn2 in zip(axes, FOLDS):
        ax.set_ylim(0.1, len(FOLDS[dn2][0]) + 0.8)
        ax.plot([], [], color="#7048e8", lw=2, label="genomic cut-off (count below)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=1, loc="upper right", bbox_to_anchor=(1.01, 1.08),
               fontsize=7.2, handletextpad=0.5, labelspacing=0.45, borderaxespad=0.2)

    label_axes(fig, axes)
    fig.tight_layout(); fig.savefig(FIG / "leakage.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)

    # ---------------- retrain once to capture R, mu, alpha ----------------
    store = {}
    for dn in FOLDS:
        z, meta = load(dn); ds = {k: z[k] for k in z.files}
        ds["U"] = np.load(ROOT / "data" / f"{dn}_interventions.npy"); gen = np.load(ROOT / "data" / f"{dn}_genomic.npz")
        T, N = ds["inc"].shape
        R = np.full((T, N), np.nan); MU = np.full((H_OUT, T, N), np.nan); Y = np.full((H_OUT, T, N), np.nan); AL = np.zeros(N)
        for f, s in enumerate(FOLDS[dn][0]):
            ds["G"] = gen[f"fold{f}"]
            for tv in (1, 4):
                p = ROOT / "data" / f"{dn}_genomic_tau{tv}.npz"
                if p.exists(): ds[f"G_tau{tv}"] = np.load(p)[f"fold{f}"]
            for fr in (25, 50, 75):
                p = ROOT / "data" / f"{dn}_genomic_f{fr}.npz"
                if p.exists(): ds[f"G_f{fr}"] = np.load(p)[f"fold{f}"]
            data = splits(ds["inc"], s, FOLDS[dn][1])
            _, _, w = train_eval(OURS, ds, data, 0, ABLATIONS[OURS])
            w.model.eval(); Xte, Yte, Mte, O = data[2]
            with torch.no_grad(): o = w.model(w.to_x(Xte), w.Uw["te"], return_diag=True)
            AL = o["phi"].numpy()
            for b, t0 in enumerate(O):
                if t0 < T: R[t0] = o["R"][b, :, 0].numpy()
                for h in range(H_OUT):
                    if Mte[b, h].any() and t0 + h < T:
                        MU[h, t0 + h] = o["mu"][b, h].numpy(); Y[h, t0 + h] = Yte[b, h].numpy()
        store[dn] = dict(inc=ds["inc"], R=R, MU=MU, Y=Y, AL=AL, meta=meta)

    # ---------------- (2) EpiEstim validation ----------------
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.1))
    agg = {}; LAGS = [-2, -1, 0, 1, 2]
    for dn, c in zip(FOLDS, ["#d9480f", "#1c7ed6"]):
        S = store[dn]; inc = S["inc"]; T, N = inc.shape
        E = np.column_stack([epiestim(inc[:, j]) for j in range(N)])
        rr = []
        for lag in LAGS:
            Es = np.roll(E, lag, axis=0)
            if lag > 0: Es[:lag] = np.nan
            if lag < 0: Es[lag:] = np.nan
            m = np.isfinite(Es) & np.isfinite(S["R"]) & (inc > 0)
            rr.append(float(spearmanr(Es[m], S["R"][m]).statistic))
        axes[0].plot(LAGS, rr, marker="o", ms=3.5, lw=1.3, color=c, label=dn)
        best = int(np.argmax(rr)); lag = LAGS[best]
        Es = np.roll(E, lag, axis=0); Es[:max(0, lag)] = np.nan
        if lag < 0: Es[lag:] = np.nan
        m = np.isfinite(Es) & np.isfinite(S["R"]) & (inc > 0)
        agg[dn] = {"best_lag_weeks": lag, "rho_at_best_lag": rr[best], "rho_at_lag0": rr[LAGS.index(0)],
                   "n": int(m.sum()),
                   "R_p5_p50_p95": [float(x) for x in np.nanpercentile(S["R"], [5, 50, 95])],
                   "epiestim_p5_p50_p95": [float(x) for x in np.nanpercentile(E, [5, 50, 95])],
                   "agree_above_below_1": float(np.mean((Es[m] > 1) == (S["R"][m] > 1)))}
        if dn == "DRC1820":
            axes[1].scatter(Es[m], S["R"][m], s=4, alpha=0.35, lw=0, c=c)
    axes[0].axhline(0, color="k", lw=0.6); axes[0].set_xticks(LAGS)
    axes[0].set_xlabel("EpiEstim lead (weeks)"); axes[0].set_ylabel("Spearman $\\rho$")
    axes[0].set_title("(a) Alignment is forward-looking", fontsize=7.6); axes[0].legend(frameon=False)
    axes[1].plot([0, 4], [0, 4], "k--", lw=0.8); axes[1].set_xlim(0, 4); axes[1].set_ylim(0, 2.6)
    axes[1].axhline(1, color="#868e96", lw=0.5); axes[1].axvline(1, color="#868e96", lw=0.5)
    axes[1].set_xlabel("EpiEstim $R_{t+1}$"); axes[1].set_ylabel("RIG-Net-GI $R_i(t)$")
    axes[1].set_title("(b) DRC1820, best lag", fontsize=7.6)
    dn = "DRC1820"; S = store[dn]; inc = S["inc"]
    j = int(np.argmax(inc.sum(0))); E = epiestim(inc[:, j]); wk = pd.to_datetime(S["meta"]["weeks"])
    ax = axes[2]
    ax.plot(wk, E, color="#1c7ed6", lw=1.0, label="EpiEstim")
    ax.plot(wk, S["R"][:, j], color="#d9480f", lw=1.5, label="RIG-Net-GI")
    ax.axhline(1, color="k", lw=0.6, ls=":"); ax.set_ylim(0, 3)
    ax.set_title("(c) " + S["meta"]["nodes"][j] + ": shrunk range", fontsize=7.6)
    ax.legend(frameon=False); ax.tick_params(axis="x", rotation=35)
    label_axes(fig, axes)
    fig.tight_layout(); fig.savefig(FIG / "epiestim.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)
    out["epiestim"] = agg

    # ---------------- (3) calibration ----------------
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.95))
    cal = {}
    qs = np.arange(0.05, 1.0, 0.05)
    for dn, c in zip(FOLDS, ["#d9480f", "#1c7ed6"]):
        S = store[dn]; mu = S["MU"]; y = S["Y"]; al = S["AL"][None, None, :]
        r = 1 / np.maximum(al, 1e-6); p = r / (r + np.nan_to_num(mu, nan=1.0))
        ok = np.isfinite(mu) & np.isfinite(y)
        emp = [float(np.mean(y[ok] <= nbinom.ppf(q, r, p)[ok])) for q in qs]
        axes[0].plot(qs, emp, marker="o", ms=2.5, lw=1.1, color=c, label=dn)
        u = nbinom.cdf(y, r, p); lo = nbinom.cdf(y - 1, r, p)
        pit = (lo + np.random.default_rng(0).random(u.shape) * (u - lo))[ok]      # randomised PIT for counts
        axes[1].hist(pit, bins=10, range=(0, 1), density=True, histtype="step", lw=1.2, color=c)
        cal[dn] = {"mean_abs_calib_err": float(np.mean(np.abs(np.array(emp) - qs))),
                   "pit_ks": float(np.max(np.abs(np.sort(pit) - np.linspace(0, 1, len(pit)))))}
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("nominal quantile"); axes[0].set_ylabel("empirical"); axes[0].set_title("Reliability", fontsize=8)
    axes[0].legend(frameon=False, fontsize=6)
    axes[1].axhline(1, color="k", ls="--", lw=0.8); axes[1].set_xlabel("randomised PIT"); axes[1].set_title("PIT", fontsize=8)
    axes[1].set_ylim(0, 2)
    label_axes(fig, axes)
    fig.tight_layout(); fig.savefig(FIG / "calibration.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)
    out["calibration"] = cal

    # ---------------- (4) critical-difference diagram ----------------
    raw = pd.read_json(RES / "results.jsonl", lines=True)
    models = BASE + [OURS]
    piv = (raw[raw.model.isin(models)].groupby(["dataset", "fold", "seed", "model"]).MAE.mean()
           .reset_index().pivot_table(index=["dataset", "fold", "seed"], columns="model", values="MAE").dropna())
    Rk = np.apply_along_axis(rankdata, 1, piv.values)
    avg = pd.Series(Rk.mean(0), index=piv.columns).sort_values()
    nD, kM = Rk.shape
    chi2, pF = friedmanchisquare(*[Rk[:, i] for i in range(kM)])
    q_alpha = 3.354                                                        # Nemenyi q for k=16, alpha=0.05
    CD = q_alpha * np.sqrt(kM * (kM + 1) / (6 * nD))
    fig, ax = plt.subplots(figsize=(7.16, 2.9))
    lo, hi = 1.0, float(kM)
    half = int(np.ceil(kM / 2))
    GAP = 0.165                      # vertical spacing between model labels
    TOP = -0.14                      # first label row below the axis
    ax.plot([lo, hi], [0, 0], "k", lw=1.1)
    for v in range(1, kM + 1):
        ax.plot([v, v], [0, 0.055], "k", lw=0.9)
        ax.text(v, 0.085, str(v), ha="center", va="bottom", fontsize=7.4)
    names = list(avg.index); vals = list(avg.values)
    for i, (m, a_) in enumerate(zip(names, vals)):
        left = i < half
        row = i if left else kM - 1 - i
        yy = TOP - GAP * row
        xe = lo - 0.55 if left else hi + 0.55
        ax.plot([a_, a_], [0, yy], color="k", lw=0.7)
        ax.plot([a_, xe], [yy, yy], color="k", lw=0.7)
        ax.text(xe + (-0.1 if left else 0.1), yy,
                f"{m} ({a_:.2f})", ha="right" if left else "left", va="center",
                fontsize=7.4, color="#d9480f" if m == OURS else "k",
                fontweight="bold" if m == OURS else "normal")
    srt = np.array(vals); drawn = 0
    for i in range(kM):
        j = i
        while j + 1 < kM and srt[j + 1] - srt[i] < CD: j += 1
        if j > i and (i == 0 or srt[j] - srt[i - 1] >= CD):
            ax.plot([srt[i] - 0.04, srt[j] + 0.04], [0.20 + 0.105 * drawn] * 2, lw=2.6, color="#495057",
                    solid_capstyle="round")
            drawn += 1
    ytop = 0.20 + 0.105 * max(drawn, 1) + 0.16
    ax.plot([lo, lo + CD], [ytop, ytop], "k", lw=1.8, solid_capstyle="butt")
    ax.plot([lo, lo], [ytop - 0.035, ytop + 0.035], "k", lw=1.2)
    ax.plot([lo + CD, lo + CD], [ytop - 0.035, ytop + 0.035], "k", lw=1.2)
    ax.text(lo + CD / 2, ytop + 0.055, f"critical distance = {CD:.2f}", ha="center", va="bottom", fontsize=7.6)
    ax.set_xlim(lo - 3.4, hi + 3.4)
    ax.set_ylim(TOP - GAP * (half - 1) - 0.18, ytop + 0.30)
    ax.axis("off")
    ax.text((lo + hi) / 2, TOP - GAP * (half - 1) - 0.16,
            f"average rank over {nD} dataset/fold/seed blocks "
            f"(Friedman $\\chi^2$={chi2:.1f}, $p$={pF:.0e}); lower is better",
            ha="center", va="top", fontsize=7.4)
    fig.tight_layout(); fig.savefig(FIG / "cd.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)
    out["cd"] = {"CD": float(CD), "n_blocks": int(nD), "friedman_p": float(pF),
                 "avg_ranks": {k: float(v) for k, v in avg.items()}}

    # ---------------- (5) variability + (7) Pareto ----------------
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.1))
    top = list(avg.index[:6])
    if "MAST-Net" not in top: top.append("MAST-Net")      # the most recent baseline, kept visible
    for col, dn in enumerate(FOLDS):
        ax = axes[col]
        g = raw[(raw.dataset == dn) & raw.model.isin(top)]
        data = [g[g.model == m].MAE.values for m in top]
        bp = ax.boxplot(data, tick_labels=[m.replace("RIG-Net-GI", "ours") for m in top], widths=0.6,
                        patch_artist=True, flierprops=dict(ms=2))
        for i, b in enumerate(bp["boxes"]):
            b.set_facecolor("#ffd8a8" if top[i] == OURS else "#dee2e6"); b.set_edgecolor("k"); b.set_lw(0.6)
        for med in bp["medians"]: med.set_color("k")
        ax.set_yscale("log"); ax.tick_params(axis="x", rotation=55, labelsize=5.8)
        ax.set_title(dn, fontsize=8.5, pad=6)
        if col == 0: ax.set_ylabel("per-run MAE (5 seeds $\\times$ folds)")
    cost = raw[raw.model.isin(models)].groupby("model").agg(sec=("train_sec", "mean"), par=("n_params", "first"))
    mae = raw[raw.model.isin(models)].groupby(["dataset", "model"]).MAE.mean().unstack(0)
    norm = (mae / mae.min()).mean(1)
    ax = axes[2]
    # numbered markers with a key, so the labels cannot collide
    named = [m for m in cost.index if m in top + ["STAEformer", "AGCRN", "Autoformer", "MAST-Net"]]
    named = sorted(named, key=lambda m: norm[m])
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as pe
    keys = []
    for q, m in enumerate(named, start=1):
        keys.append(Line2D([], [], ls="none", marker="$%d$" % q, color="#495057", ms=5,
                           label=m.replace("RIG-Net-GI", "ours")))
    ax.scatter(cost.sec, norm, s=42, c=["#d9480f" if m == OURS else "#74c0fc" for m in cost.index],
               edgecolors="k", linewidths=0.3, zorder=3)
    for q, m in enumerate(named, start=1):
        ax.annotate(str(q), (cost.sec[m], norm[m]), fontsize=5.2, ha="center", va="center",
                    zorder=4, color="#10243f",
                    path_effects=[pe.withStroke(linewidth=1.3, foreground="white")])
    ax.set_ylim(norm.min() * 0.97, norm.max() * 1.42)      # headroom for the key
    ax.legend(handles=keys, frameon=False, fontsize=5.0, ncol=3, loc="upper center",
              handletextpad=0.1, columnspacing=0.7, labelspacing=0.22, borderpad=0.1,
              bbox_to_anchor=(0.5, 1.0))
    ax.set_xscale("log"); ax.set_xlabel("train seconds / fold-seed"); ax.set_ylabel("mean normalised MAE")
    ax.set_title("Accuracy vs cost", fontsize=8)
    label_axes(fig, axes)
    fig.tight_layout(); fig.savefig(FIG / "variability.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)

    # ---------------- (6) dose-response + graph sensitivity ----------------
    dose = raw[raw.model.str.contains("gen")].copy()
    geo = raw[raw.model.str.contains("geo s")].copy()
    if len(dose):
        fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.95))
        for ax, dn in zip(axes, FOLDS):
            d = dose[dose.dataset == dn].groupby("model").MAE.mean()
            base = raw[(raw.model == OURS) & (raw.dataset == dn)].MAE.mean()
            nog = raw[(raw.model == "RIG-Net-GI w/o genomic prior") & (raw.dataset == dn)].MAE.mean()
            xs = [0, 25, 50, 75, 100]
            ys = [nog] + [d.get(f"RIG-Net-GI gen{f}%", np.nan) for f in (25, 50, 75)] + [base]
            ax.plot(xs, ys, marker="o", ms=3.5, color="#7048e8", lw=1.3)
            ax.set_title(dn, fontsize=8.5, pad=6); ax.set_xlabel("% of genomes used", fontsize=7.4)
        axes[0].set_ylabel("test MAE", fontsize=7.4)
        label_axes(fig, axes)
        fig.tight_layout(); fig.savefig(FIG / "dose.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)
        out["dose"] = {f"{a}|{b}": float(v) for (a, b), v in dose.groupby(["dataset", "model"]).MAE.mean().items()}
    if len(geo):
        out["geo_sensitivity"] = {f"{a}|{b}": float(v) for (a, b), v in geo.groupby(["dataset", "model"]).MAE.mean().items()}

    json.dump(out, open(OUT / "paper_numbers_valid.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "cd"}, indent=1, default=str)[:2500])
    print("avg ranks:", {k: round(v, 2) for k, v in out["cd"]["avg_ranks"].items()})


if __name__ == "__main__":
    main()
