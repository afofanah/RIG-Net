"""analyze.py - aggregate rolling-origin results, significance tests, figures."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from baselines import BASELINES
from run_experiments import FOLDS, ABLATIONS

ROOT = Path(__file__).parent; RES = ROOT / "results"; FIG = RES / "figures"; FIG.mkdir(parents=True, exist_ok=True)
fn = lambda m: m.replace(" ", "_").replace("/", "")
REF = ["Naive", "MA3"]; BASE = list(BASELINES); OURS = "RIG-Net"; ABL = [a for a in ABLATIONS if a != OURS]
PHASE = {"WA2014": ["Peak (Oct-Nov 14)", "Decline (Nov-Dec 14)", "Tail (Jan-Feb 15)"],
         "DRC1820": ["Growth (Apr-Jul 19)", "Plateau (Jul-Sep 19)", "Decline (Sep-Dec 19)", "Tail (Dec 19-Mar 20)"]}


def pooled(dn, model, seeds):
    """Concatenate test points over folds -> per-seed arrays (pred, y, horizon, fold)."""
    out = []
    for s in seeds:
        P, Y, Hh, Fd = [], [], [], []
        for f in range(len(FOLDS[dn][0])):
            d = RES / "preds" / dn / f"fold{f}"; p = d / f"{fn(model)}__s{s}.npy"
            if not p.exists(): return None
            pr, y, m = np.load(p), np.load(d / "Y.npy"), np.load(d / "M.npy")
            h = np.broadcast_to(np.arange(y.shape[1])[None, :, None], y.shape)
            P.append(pr[m]); Y.append(y[m]); Hh.append(h[m]); Fd.append(np.full(m.sum(), f))
        out.append((np.concatenate(P), np.concatenate(Y), np.concatenate(Hh), np.concatenate(Fd)))
    return out


def scores(p, y):
    mae = np.abs(p - y).mean(); rmse = np.sqrt(((p - y) ** 2).mean())
    pcc = np.corrcoef(p, y)[0, 1] if p.std() > 0 else 0.0
    return mae, rmse, pcc


def dm_test(e1, e2):
    """Diebold-Mariano on absolute-error loss differential (positive stat => model 2 better)."""
    d = np.abs(e1) - np.abs(e2); n = len(d); g0 = d.var()
    stat = d.mean() / np.sqrt(g0 / n + 1e-12); return stat, 2 * (1 - norm.cdf(abs(stat)))


def holm(pvals):
    idx = np.argsort(pvals); adj = np.empty_like(pvals); m = len(pvals); run = 0
    for r, i in enumerate(idx): run = max(run, min(1, (m - r) * pvals[i])); adj[i] = run
    return adj


def main():
    raw = pd.read_json(RES / "results.jsonl", lines=True)
    seeds = sorted(raw[raw.model == OURS].seed.unique())
    tables, sig_rows, horizon_rows, phase_rows = {}, [], [], []
    for dn in FOLDS:
        rows = []
        cache = {}
        for m in REF + BASE + [OURS] + ABL:
            ss = [0] if m in REF else seeds
            pl = pooled(dn, m, ss)
            if pl is None: continue
            cache[m] = pl
            sc = np.array([scores(p, y) for p, y, _, _ in pl])
            fam = "reference" if m in REF else ("ours" if m == OURS else ("post-hoc" if "post-hoc" in m else ("ablation" if m in ABL else BASELINES[m][3])))
            yr = "" if m not in BASELINES else f"{BASELINES[m][2]} {BASELINES[m][1]}"
            rows.append({"Model": m, "Venue": yr, "Family": fam,
                         "MAE": sc[:, 0].mean(), "MAE_sd": sc[:, 0].std(), "RMSE": sc[:, 1].mean(), "RMSE_sd": sc[:, 1].std(),
                         "PCC": sc[:, 2].mean(), "PCC_sd": sc[:, 2].std(), "n_points": len(pl[0][0])})
            # per-horizon & per-phase (seed-averaged prediction)
            pm = np.mean([p for p, _, _, _ in pl], 0); y, hh, fd = pl[0][1], pl[0][2], pl[0][3]
            for h in range(4):
                horizon_rows.append({"dataset": dn, "model": m, "h": h + 1, "MAE": np.abs(pm - y)[hh == h].mean()})
            for f in np.unique(fd):
                phase_rows.append({"dataset": dn, "model": m, "phase": PHASE[dn][f], "MAE": np.abs(pm - y)[fd == f].mean()})
        t = pd.DataFrame(rows); tables[dn] = t
        # significance: RIG-Net (seed-avg) vs every baseline/reference (seed-avg), paired over test points
        if OURS in cache:
            ours = np.mean([p for p, _, _, _ in cache[OURS]], 0); y = cache[OURS][0][1]; e0 = ours - y
            comp = [m for m in REF + BASE if m in cache]; pw, dms = [], []
            for m in comp:
                e1 = np.mean([p for p, _, _, _ in cache[m]], 0) - y
                diff = np.abs(e1) - np.abs(e0)
                pw.append(wilcoxon(np.abs(e1), np.abs(e0), zero_method="zsplit").pvalue if np.any(diff != 0) else 1.0)
                dms.append(dm_test(e1, e0))
            adj = holm(np.array(pw))
            for m, p, a, (st, pd_) in zip(comp, pw, adj, dms):
                e1 = np.mean([p_ for p_, _, _, _ in cache[m]], 0) - y
                sig_rows.append({"dataset": dn, "baseline": m, "MAE_baseline(seed-avg)": np.abs(e1).mean(),
                                 "MAE_RIG-Net(seed-avg)": np.abs(e0).mean(), "Wilcoxon_p": p, "Holm_p": a,
                                 "DM_stat": st, "DM_p": pd_, "RIG-Net_better": np.abs(e0).mean() < np.abs(e1).mean()})
    pd.concat([t.assign(Dataset=k) for k, t in tables.items()]).to_csv(RES / "main_table.csv", index=False)
    pd.DataFrame(sig_rows).to_csv(RES / "significance.csv", index=False)
    hz = pd.DataFrame(horizon_rows); hz.to_csv(RES / "per_horizon.csv", index=False)
    ph = pd.DataFrame(phase_rows); ph.to_csv(RES / "per_phase.csv", index=False)

    for dn, t in tables.items():
        print(f"\n===== {dn}  (seeds={len(seeds)})")
        tt = t.copy(); tt["rank"] = tt[tt.Family.isin(["temporal", "graph", "ours", "reference"])].MAE.rank()
        print(tt.sort_values("MAE")[["Model", "Family", "Venue", "MAE", "MAE_sd", "RMSE", "PCC", "rank"]].round(3).to_string(index=False))
    if sig_rows:
        print("\n", pd.DataFrame(sig_rows).round(4).to_string(index=False))

    # ---------------- figures
    for dn, t in tables.items():
        tm = t[~t.Family.isin(["ablation", "post-hoc"])].sort_values("MAE")
        col = {"reference": "#9aa0a6", "temporal": "#5b8def", "graph": "#2bb39a", "ours": "#e8590c"}
        fig, ax = plt.subplots(figsize=(7, 5.2))
        ax.barh(tm.Model, tm.MAE, xerr=tm.MAE_sd, color=[col[f] for f in tm.Family])
        ax.invert_yaxis(); ax.set_xlabel("MAE (weekly cases per region), pooled over rolling-origin folds")
        ax.set_title(f"{dn}: test MAE (mean ± sd over seeds)"); fig.tight_layout(); fig.savefig(FIG / f"mae_bar_{dn}.png", dpi=160); plt.close(fig)
        h = hz[hz.dataset == dn]
        best_bases = t[t.Family.isin(["temporal", "graph"])].nsmallest(4, "MAE").Model.tolist()
        fig, ax = plt.subplots(figsize=(6, 4))
        for m in ["Naive"] + best_bases + [OURS]:
            hh = h[h.model == m]; ax.plot(hh.h, hh.MAE, marker="o", lw=2.5 if m == OURS else 1.3, label=m,
                                          color="#e8590c" if m == OURS else None)
        ax.set_xlabel("Forecast horizon (weeks)"); ax.set_ylabel("MAE"); ax.set_xticks([1, 2, 3, 4]); ax.legend(fontsize=8)
        ax.set_title(f"{dn}: error by horizon"); fig.tight_layout(); fig.savefig(FIG / f"horizon_{dn}.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
