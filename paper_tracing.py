"""paper_tracing.py - the onward-transmission network the model predicts, per country.

For every test week the decoder states how much of region j's next-week pressure is drawn
from region i:  F_{i->j} = R_j * rho_j * A_{ji} * P_i,  in expected cases. Summing F over the
test weeks gives a directed, weighted graph of where the model expects transmission to travel,
which is the quantity a tracing team would prioritise on. It is a model output, not a record of
observed contacts: no contact-tracing data exists for these outbreaks at this resolution.
"""
import json, sys
from pathlib import Path
from paths import FIG, PAPER
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from panels import label_axes

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import FOLDS, ABLATIONS, load, splits, train_eval, H_OUT, ROOT
from geobase import draw as draw_countries

OUT = PAPER; FIG = OUT / "figs"
OURS = "RIG-Net-GI"
W_GI = np.array([0.185, 0.361, 0.258, 0.135, 0.061])
CC = {"GIN": "Guinea", "LBR": "Liberia", "SLE": "Sierra Leone", "COD": "DR Congo"}
COL = {"GIN": "#1c7ed6", "LBR": "#2f9e44", "SLE": "#862e9c", "COD": "#d9480f"}
plt.rcParams.update({"savefig.dpi": 400, "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.6,
                     "axes.spines.top": False, "axes.spines.right": False})


def flows(dn):
    """Accumulate F_{i->j} (expected imported cases) over all test weeks of every fold."""
    z, meta = load(dn); ds = {k: z[k] for k in z.files}
    ds["U"] = np.load(ROOT / "data" / f"{dn}_interventions.npy")
    gen = np.load(ROOT / "data" / f"{dn}_genomic.npz")
    inc = ds["inc"]; T, N = inc.shape
    F = np.zeros((N, N)); out_mu = np.zeros(N); Rbar = np.zeros(N); nseen = np.zeros(N)
    Fin = np.full((T, N), np.nan); Fout = np.full((T, N), np.nan)   # per-week import / export pressure
    rho_acc = np.zeros(N)
    for f, s in enumerate(FOLDS[dn][0]):
        ds["G"] = gen[f"fold{f}"]
        for tag, pat, vals in (("G_tau%d", "{dn}_genomic_tau{v}.npz", (1, 4)),
                               ("G_f%d", "{dn}_genomic_f{v}.npz", (25, 50, 75))):
            for v in vals:
                p = ROOT / "data" / pat.format(dn=dn, v=v)
                if p.exists(): ds[tag % v] = np.load(p)[f"fold{f}"]
        data = splits(ds["inc"], s, FOLDS[dn][1])
        _, _, w = train_eval(OURS, ds, data, 0, ABLATIONS[OURS])
        w.model.eval(); Xte, Yte, Mte, O = data[2]
        with torch.no_grad(): o = w.model(w.to_x(Xte), w.Uw["te"], return_diag=True)
        A = o["A"].numpy(); rho = o["rho"].numpy(); R = o["R"][..., 0].numpy()
        for b, t0 in enumerate(O):
            if t0 >= T: continue
            P = sum(W_GI[k] * inc[max(t0 - k - 1, 0)] for k in range(5))          # [N]
            Ab = A[b] if A.ndim == 3 else A
            # expected cases in j drawn from i next week
            Fw_ = (R[b] * rho[b])[:, None] * Ab * P[None, :]       # [j, i] cases j imports from i
            F += Fw_
            Fin[t0] = Fw_.sum(1); Fout[t0] = Fw_.sum(0)
            out_mu += R[b] * ((1 - rho[b]) * P + rho[b] * (Ab @ P))
            Rbar += R[b]; rho_acc += rho[b]; nseen += 1
    Rbar = np.divide(Rbar, np.maximum(nseen, 1))
    rho_m = np.divide(rho_acc, np.maximum(nseen, 1))
    Glast = gen[f"fold{len(FOLDS[dn][0])-1}"]
    return (F.T, out_mu, Rbar, ds["coords"], meta, Fin, Fout, inc,
            rho_m, ds["W"], Glast, (Glast.sum(1) > 0))


def panel(ax, idx, F, mu, Rbar, coords, names, title, colour, Wgeo, G, seq, topk=12,
          legloc="upper left", panel_key=None):
    """One country: three edge types over the same nodes, markers by sequencing status."""
    sub = F[np.ix_(idx, idx)].copy(); np.fill_diagonal(sub, 0.0)
    co = coords[idx].astype(float).copy(); mm = mu[idx]; rr = Rbar[idx]
    # gentle repulsion so co-located seats separate without distorting the map
    span = max(np.ptp(co[:, 1]), np.ptp(co[:, 0]), 1e-6); rmin = 0.085 * span
    for _ in range(220):
        d = co[:, None, :] - co[None, :, :]
        dist = np.hypot(d[..., 0], d[..., 1]); np.fill_diagonal(dist, np.inf)
        close = dist < rmin
        if not close.any(): break
        push = np.zeros_like(co)
        ii, jj = np.where(close)
        for a_, b_ in zip(ii, jj):
            v = co[a_] - co[b_]; n = np.hypot(*v)
            v = v / n if n > 1e-9 else np.random.default_rng(a_ + b_).normal(size=2)
            push[a_] += 0.5 * (rmin - n) * v
        co = co + 0.35 * push
    nm = [names[i].split(":")[-1].replace("Western Area", "W. Area") for i in idx]
    Wg = Wgeo[np.ix_(idx, idx)]; Gg = G[np.ix_(idx, idx)]; sq = seq[idx]

    # (r1) geographic adjacency
    for a_ in range(len(idx)):
        for b_ in range(a_ + 1, len(idx)):
            if Wg[a_, b_] > 0 or Wg[b_, a_] > 0:
                ax.plot([co[a_, 1], co[b_, 1]], [co[a_, 0], co[b_, 0]],
                        color="#ced4da", lw=0.7, alpha=0.7, zorder=1)
    # (r3) genomic transmission linkage
    gvals = Gg[np.triu_indices(len(idx), 1)]
    gthr = np.sort(gvals[gvals > 0])[-min(18, max((gvals > 0).sum(), 1))] if (gvals > 0).any() else np.inf
    for a_ in range(len(idx)):
        for b_ in range(a_ + 1, len(idx)):
            if Gg[a_, b_] >= gthr and Gg[a_, b_] > 0:
                ax.plot([co[a_, 1], co[b_, 1]], [co[a_, 0], co[b_, 0]], color="#e03131",
                        lw=0.5 + 1.3 * min(Gg[a_, b_] / max(Gg.max(), 1e-9), 1), ls=":",
                        alpha=0.65, zorder=2)
    # (r2) predicted transmission / tracing edges
    if sub.size and sub.max() > 0:
        thr = np.sort(sub.ravel())[-min(topk, sub.size)]
        for a_ in range(len(idx)):
            for b_ in range(len(idx)):
                if a_ != b_ and sub[a_, b_] >= thr and sub[a_, b_] > 0:
                    ax.annotate("", xy=(co[b_, 1], co[b_, 0]), xytext=(co[a_, 1], co[a_, 0]),
                                arrowprops=dict(arrowstyle="-|>", color="#5f3dc4", alpha=0.85,
                                                lw=0.7 + 2.6 * sub[a_, b_] / sub.max(),
                                                connectionstyle="arc3,rad=0.13",
                                                shrinkA=7, shrinkB=7), zorder=4)
    sz = 26 + 300 * mm / max(mm.max(), 1e-9)
    for mk, mask, lab in (("o", sq, "sequenced"), ("^", ~sq, "no genomes")):
        if mask.any():
            sc = ax.scatter(co[mask, 1], co[mask, 0], s=sz[mask], c=rr[mask], marker=mk,
                            cmap="RdYlBu_r", vmin=0.4, vmax=1.6, edgecolors="black",
                            linewidths=0.6, zorder=5)
    # top sources are numbered on the map and named in a corner legend, so no label overlaps
    rank = np.argsort(-sub.sum(1))[:3]
    from matplotlib.lines import Line2D
    keys = []
    for q, a_ in enumerate(rank, start=1):
        import matplotlib.patheffects as pe
        ax.annotate(str(q), (co[a_, 1], co[a_, 0]), fontsize=7.8, fontweight="bold",
                    color="#10243f", ha="center", va="center", zorder=7,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
        keys.append(Line2D([], [], marker="o", ls="none", ms=7.5, markerfacecolor="none",
                           markeredgecolor="#1864ab", markeredgewidth=1.6,
                           label=f"{q}.\u2009{nm[a_]}  $\\rho$ {rho_disp[idx[a_]]:.2f}"))
        ax.scatter([co[a_, 1]], [co[a_, 0]], s=150, facecolors="none",
                   edgecolors="#1864ab", linewidths=1.6, zorder=6)
    ax.legend(handles=keys, frameon=True, framealpha=0.92, edgecolor="#adb5bd",
              fontsize=7.2, loc=legloc, handletextpad=0.4, borderpad=0.4,
              labelspacing=0.3).set_zorder(8)

    ax.margins(x=0.17, y=0.24)
    if panel_key: draw_countries(ax, panel_key)
    ax.set_title(title, color=colour, fontsize=11)
    ax.set_xlabel("Longitude", fontsize=9); ax.margins(x=0.17, y=0.24); ax.tick_params(labelsize=8.4)
    return sc, [nm[a_] for a_ in rank], sub


def main():
    res = {}
    (Fw, muw, Rw, cow, mw, Finw, Foutw, incw, rhow, Wgw, Gw, sqw) = flows("WA2014")
    (Fd, mud, Rd, cod, md, Find, Foutd, incd, rhod, Wgd, Gd, sqd) = flows("DRC1820")
    cc = np.array([n.split(":")[0] for n in mw["nodes"]])

    global rho_disp
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="#adb5bd", lw=1.4, label="geographic adjacency"),
               Line2D([], [], color="#e03131", lw=1.4, ls=":", label="genomic linkage"),
               Line2D([], [], color="#5f3dc4", lw=2.2, label="predicted transmission (tracing edge)"),
               Line2D([], [], marker="o", color="none", markerfacecolor="#ffd8a8",
                      markeredgecolor="k", ms=7, label="region, sequenced"),
               Line2D([], [], marker="^", color="none", markerfacecolor="#ffd8a8",
                      markeredgecolor="k", ms=7, label="region, no genomes"),
               Line2D([], [], marker="o", color="none", markerfacecolor="#dee2e6",
                      markeredgecolor="k", ms=10, label="marker area = predicted incidence")]

    for part, codes, legloc in (("a", ("GIN", "LBR"), "upper right"),
                                ("b", ("SLE", "COD"), "upper left")):
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.9),
                                 gridspec_kw={"wspace": 0.18})
        for ax, code in zip(axes, codes):
            if code == "COD":
                idx = np.arange(len(cod)); rho_disp = rhod
                sc, top, sub = panel(ax, idx, Fd, mud, Rd, cod, md["nodes"], CC[code], COL[code],
                                     Wgd, Gd, sqd, topk=12, legloc=legloc, panel_key='DRC1820')
                tot, cross = float(Fd.sum()), 0.0
            else:
                idx = np.where(cc == code)[0]; rho_disp = rhow
                sc, top, sub = panel(ax, idx, Fw, muw, Rw, cow, mw["nodes"], CC[code], COL[code],
                                     Wgw, Gw, sqw, topk=12, legloc=legloc, panel_key='WA2014')
                tot = float(Fw[np.ix_(idx, np.arange(len(cow)))].sum())
                cross = float(Fw[np.ix_(idx, np.where(cc != code)[0])].sum())
            res.setdefault(CC[code], {}).update(
                {"top_sources": top, "within_country_expected_cases": float(sub.sum()),
                 "total_outgoing": tot, "cross_border_share": (cross / tot) if tot > 0 else 0.0})
            ax.set_ylabel("Latitude" if ax is axes[0] else "", fontsize=9)
        cb = fig.colorbar(sc, ax=axes, pad=0.012, fraction=0.024)
        cb.set_label("learned $R_i$", fontsize=8.5); cb.ax.tick_params(labelsize=7.6)
        fig.legend(handles=handles, frameon=False, ncol=3, loc="lower center",
                   bbox_to_anchor=(0.5, -0.14), fontsize=8.2)
        label_axes(fig, axes)
        fig.savefig(FIG / f"tracing_{part}.pdf", bbox_inches="tight", pad_inches=0.015)
        plt.close(fig)

    # ======== time-step view and spatio-temporal correlation ========
    from scipy.stats import pearsonr, spearmanr
    panels = [("GIN", Finw, Foutw, incw, mw, cc), ("LBR", Finw, Foutw, incw, mw, cc),
              ("SLE", Finw, Foutw, incw, mw, cc),
              ("COD", Find, Foutd, incd, md, np.array(["COD"] * incd.shape[1]))]
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.1),
                             gridspec_kw={"height_ratios": [1.45, 1.0], "hspace": 0.48,
                                          "wspace": 0.72})
    lags = np.arange(-3, 4)
    for col, (code, Fin_, Fout_, inc_, meta_, ccv) in enumerate(panels):
        idx = np.where(ccv == code)[0]
        cov = np.isfinite(Fin_[:, idx]).any(1)
        sub = Fin_[np.ix_(cov, idx)]; obs = inc_[np.ix_(cov, idx)]

        # (top) total predicted import pressure per region, sorted
        tot = np.nansum(sub, 0)
        order = np.argsort(tot)[-6:]
        def short(i):
            n = meta_["nodes"][idx[i]].split(":")[-1]
            n = n.replace("Western Area", "W. Area").replace("Grand ", "Gr. ")
            return n if len(n) <= 13 else n[:12] + "."
        obs_tot = np.nansum(obs, 0)                       # observed cases, the only reference
        rho_rank = spearmanr(tot, obs_tot).statistic if len(tot) > 2 else np.nan
        ax = axes[0, col]
        yy = np.arange(len(order)); h = 0.38
        ax.barh(yy + h / 2, tot[order] / max(tot.max(), 1e-9), h, color=COL[code], alpha=0.9,
                edgecolor="black", linewidth=0.3, label="predicted imported")
        ax.barh(yy - h / 2, obs_tot[order] / max(obs_tot.max(), 1e-9), h, color="#adb5bd",
                edgecolor="black", linewidth=0.3, label="observed cases")
        ax.set_yticks(yy); ax.set_yticklabels([short(o) for o in order], fontsize=6.4)
        ax.set_title(CC[code], color=COL[code], fontsize=9)
        ax.set_xlabel("share of panel maximum", fontsize=7.4)
        ax.set_xlim(0, 1.18); ax.tick_params(axis="x", labelsize=6.8)
        ax.text(0.98, 0.04, f"$\\rho_s$={rho_rank:.2f}", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=6.8, color="#495057",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
        if col == 0:
            ax.legend(frameon=False, fontsize=6.4, loc="lower center",
                      bbox_to_anchor=(2.35, 1.22), ncol=2)
        res.setdefault(CC[code], {})["rank_corr_with_observed"] = float(rho_rank)

        # (bottom) lagged correlation: does predicted import pressure lead observed cases?
        ax2 = axes[1, col]; rs = []
        for L in lags:
            a_, b_ = [], []
            for q in range(sub.shape[1]):
                x = sub[:, q]; y = np.roll(obs[:, q], -L)
                m = np.isfinite(x) & np.isfinite(y)
                if L > 0: m[-L:] = False
                if L < 0: m[:-L] = False
                if m.sum() > 4: a_.append(x[m]); b_.append(y[m])
            rs.append(pearsonr(np.concatenate(a_), np.concatenate(b_))[0] if a_ else np.nan)
        rs = np.array(rs)
        ax2.bar(lags, rs, color=COL[code], alpha=0.85, width=0.65)
        ax2.axhline(0, color="k", lw=0.6); ax2.set_xticks(lags)
        ax2.set_xlabel("lag (weeks)", fontsize=8.5); ax2.set_ylim(min(0, np.nanmin(rs) * 1.2), 0.70)
        if col == 0: ax2.set_ylabel("corr. with observed", fontsize=8.5)
        ax2.tick_params(labelsize=7.2)
        best = int(lags[np.nanargmax(rs)])
        res.setdefault(CC[code], {}).update(
            {"best_lag_weeks": best, "corr_at_best_lag": float(np.nanmax(rs)),
             "corr_at_lag0": float(rs[list(lags).index(0)])})
    label_axes(fig, axes, align="left")
    fig.savefig(FIG / "tracing_time.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)

    json.dump(res, open(OUT / "paper_numbers_tracing.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
