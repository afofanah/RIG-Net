"""diagnostics.py - interpretability of RIG-Net + forecast trajectories."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import FOLDS, ABLATIONS, load, splits, train_eval, H_OUT

ROOT = Path(__file__).parent; RES = ROOT / "results"; FIG = RES / "figures"
fn = lambda m: m.replace(" ", "_").replace("/", "")


def collect(dn, seed=0):
    z, meta = load(dn); ds = {k: z[k] for k in z.files}; st, b = FOLDS[dn]; rows, gates = [], []
    for f, s in enumerate(st):
        data = splits(ds["inc"], s, b)
        _, _, w = train_eval("RIG-Net", ds, data, seed, ABLATIONS["RIG-Net"])
        Xte, _, Mte, Ote = data[2]
        w.model.eval()
        with torch.no_grad():
            o = w.model(w.to_x(Xte), return_diag=True)
        for bi, t in enumerate(Ote):
            if t < s: continue                                   # origin inside test block
            for j, node in enumerate(meta["nodes"]):
                rows.append({"week": meta["weeks"][t], "node": node, "R_h1": float(o["R"][bi, j, 0]),
                             "rho": float(o["rho"][bi, j]), "iota": float(o["iota"][bi, j]),
                             "obs_prev": float(Xte[bi, -1, j])})
        gates.append(o["pi"].mean((0, 1)).numpy())
    g = np.mean(gates, 0)
    return pd.DataFrame(rows), {"geo": float(g[0]), "adaptive": float(g[1]), "dynamic": float(g[2])}, meta, ds


def forecast_plot(dn, meta, ds, best_base, nodes_top=4):
    inc = ds["inc"]; st, b = FOLDS[dn]; tot = inc.sum(0); top = np.argsort(-tot)[:nodes_top]
    fig, axes = plt.subplots(nodes_top, 1, figsize=(9, 2.2 * nodes_top), sharex=True)
    weeks = pd.to_datetime(meta["weeks"])
    for ax, j in zip(axes, top):
        ax.plot(weeks, inc[:, j], color="k", lw=1.2, label="observed")
        for m, c in [("RIG-Net", "#e8590c"), (best_base, "#5b8def"), ("Naive", "#9aa0a6")]:
            for f, s in enumerate(st):
                d = RES / "preds" / dn / f"fold{f}"; O = np.load(d / "O.npy")
                files = sorted(d.glob(f"{fn(m)}__s*.npy"))
                if not files: continue
                P = np.mean([np.load(x) for x in files], 0)
                sel = (O >= s) & (O < s + b) & (O + 1 < len(weeks))
                ax.plot(weeks[O[sel] + 1], P[sel, 1, j], color=c, lw=1.4, ls="-" if m == "RIG-Net" else "--",
                        label=f"{m} (2-wk ahead)" if f == 0 else None)
        for s in st: ax.axvspan(weeks[s], weeks[min(s + b, len(weeks)) - 1], color="#fff3e0", zorder=0)
        ax.set_ylabel(meta["nodes"][j].split(":")[-1], fontsize=8)
    axes[0].legend(fontsize=7, ncol=4); axes[0].set_title(f"{dn}: 2-week-ahead forecasts in rolling test blocks (shaded)")
    fig.tight_layout(); fig.savefig(FIG / f"forecasts_{dn}.png", dpi=160); plt.close(fig)


def main():
    table = pd.read_csv(RES / "main_table.csv"); out = {}
    for dn in FOLDS:
        t = table[(table.Dataset == dn) & table.Family.isin(["temporal", "graph"])]
        best = t.nsmallest(1, "MAE").Model.iloc[0]
        df, gates, meta, ds = collect(dn); df.to_csv(RES / f"rignet_diagnostics_{dn}.csv", index=False)
        forecast_plot(dn, meta, ds, best)
        top = ds["inc"].sum(0).argsort()[::-1][:8]; topn = [meta["nodes"][i] for i in top]
        piv = df[df.node.isin(topn)].pivot_table(index="node", columns="week", values="R_h1").loc[topn]
        fig, ax = plt.subplots(figsize=(10, 3.6))
        im = ax.imshow(piv.values, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=2.5)
        ax.set_yticks(range(len(topn))); ax.set_yticklabels([n.split(":")[-1] for n in topn], fontsize=8)
        ax.set_xticks(range(0, piv.shape[1], 3)); ax.set_xticklabels(piv.columns[::3], rotation=60, fontsize=7)
        plt.colorbar(im, label="RIG-Net R (1-wk ahead)"); ax.set_title(f"{dn}: learned regional reproduction number in test blocks")
        fig.tight_layout(); fig.savefig(FIG / f"R_heatmap_{dn}.png", dpi=160); plt.close(fig)
        grp = df.assign(group=df.node.str.split(":").str[0] if dn == "WA2014" else "DRC").groupby("group")
        out[dn] = {"graph_gate_share": gates, "best_baseline": best,
                   "rho_by_group": grp.rho.mean().round(3).to_dict(),
                   "R_median_active": float(df[df.obs_prev > 0].R_h1.median()),
                   "R_median_inactive": float(df[df.obs_prev == 0].R_h1.median())}
        print(dn, json.dumps(out[dn], indent=1))
    json.dump(out, open(RES / "rignet_interpretability.json", "w"), indent=1)


if __name__ == "__main__":
    main()
