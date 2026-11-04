"""paper_seqintegration.py - how genomic sequencing was integrated into the DRC response.

(a) weekly cases split into sequenced and unsequenced, from genome collection dates;
(b) genomes against cases per health zone, the proportionality the prior's normalisation assumes;
(c) the one delay the metadata supports, symptom onset to sample collection.

Sequencing and release dates are not in the public metadata, so the later steps of the
sample-to-sequence pipeline cannot be measured here and are not shown.
"""
import json, re, sys, unicodedata
from pathlib import Path
from paths import FIG, PAPER
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from panels import label_axes

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import load, ROOT

OUT = PAPER; FIG = OUT / "figs"
SEQ, UNSEQ, ACC = "#f59f00", "#ced4da", "#d9480f"
plt.rcParams.update({"savefig.dpi": 400, "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2,
                     "axes.spines.top": False, "axes.spines.right": False})


def norm(s):
    return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", str(s))
                  .encode("ascii", "ignore").decode().lower())


def main():
    m = pd.read_csv(ROOT / "external" / "drc_metadata.tsv", sep="\t")
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m["onset"] = pd.to_datetime(m["date_symptom_onset"], errors="coerce")
    z, meta = load("DRC1820")
    inc = z["inc"]; weeks = pd.to_datetime(meta["weeks"])
    nodes = [n.split(":")[-1] for n in meta["nodes"]]
    out = {}

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5),
                             gridspec_kw={"width_ratios": [1.9, 1.0, 1.0], "wspace": 0.34})

    # ---------- (a) sequenced vs unsequenced cases, week by week ----------
    seq_w = np.zeros(len(weeks))
    idx = pd.Series(range(len(weeks)), index=weeks)
    for d in m["date"].dropna():
        j = idx.index.searchsorted(d, side="right") - 1
        if 0 <= j < len(seq_w): seq_w[j] += 1
    tot_w = inc.sum(1)
    seq_w = np.minimum(seq_w, tot_w)                      # genomes cannot exceed reported cases
    ax = axes[0]
    ax.bar(weeks, seq_w, width=pd.Timedelta(days=6), color=SEQ, label="sequenced")
    ax.bar(weeks, tot_w - seq_w, width=pd.Timedelta(days=6), bottom=seq_w, color=UNSEQ,
           label="not sequenced")
    # the peak window, where coverage collapses
    lo, hi = pd.Timestamp("2019-03-24"), pd.Timestamp("2019-07-07")
    w = (weeks >= lo) & (weeks <= hi)
    cov_pk = 100 * seq_w[w].sum() / max(tot_w[w].sum(), 1)
    cov_el = 100 * seq_w[~w].sum() / max(tot_w[~w].sum(), 1)
    ax.axvspan(lo, hi, color="#e03131", alpha=0.07, zorder=0)
    for b in (lo, hi):
        ax.axvline(b, color="#e03131", lw=0.7, ls=":", zorder=1)
    ax.annotate(f"epidemic peak:\n{cov_pk:.0f}% sequenced", xy=(lo + (hi - lo) / 2, tot_w.max() * 0.78),
                ha="center", va="top", fontsize=6.8, color="#c92a2a")
    ax.annotate(f"{cov_el:.0f}% elsewhere", xy=(weeks[int(len(weeks) * 0.70)], tot_w.max() * 0.50),
                ha="left", fontsize=6.8, color="#495057")
    out["peak_window_coverage"] = float(cov_pk); out["offpeak_coverage"] = float(cov_el)
    ax.set_ylabel("weekly cases"); ax.set_title("Sequencing coverage over time")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.tick_params(axis="x", rotation=30)
    out["pct_sequenced"] = float(100 * seq_w.sum() / max(tot_w.sum(), 1))
    out["peak_week_coverage"] = float(100 * seq_w[int(np.argmax(tot_w))] / max(tot_w.max(), 1))

    # ---------- (b) genomes against cases, per health zone ----------
    cnt = m.groupby(m.health_zone.map(norm)).size()
    x = inc.sum(0); y = np.array([cnt.get(norm(n), 0) for n in nodes], float)
    ax = axes[1]
    ax.scatter(x, y, s=22, c=ACC, alpha=0.8, edgecolors="k", linewidths=0.3)
    lim = [0, max(x.max(), 1) * 1.08]
    b = np.polyfit(x, y, 1)
    ax.plot(lim, np.polyval(b, lim), "k:", lw=0.9)
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    resid = y - np.polyval(b, x)
    for i, (dx, dy, ha) in zip([int(np.argmin(resid)), int(np.argmax(resid))],
                               [(-5, -11, "right"), (5, 5, "left")]):
        ax.annotate(nodes[i], (x[i], y[i]), fontsize=6.4, color="#495057", ha=ha,
                    xytext=(dx, dy), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color="#adb5bd", lw=0.5, shrinkA=0, shrinkB=2))
    zero = [n for n, yy in zip(nodes, y) if yy == 0]
    ax.text(0.04, 0.93, f"{len(zero)} zones with no genomes\n(at most {int(max(x[y == 0]))} cases each)",
            transform=ax.transAxes, fontsize=6.2, color="#495057", va="top")
    ax.text(0.95, 0.08, f"$R^2$={r2:.2f}", transform=ax.transAxes, ha="right", fontsize=7.6)
    out["most_undersequenced"] = nodes[int(np.argmin(resid))]
    out["most_oversequenced"] = nodes[int(np.argmax(resid))]
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(200)); ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.tick_params(axis="x", labelsize=7.0)
    ax.set_xlabel("cases in the zone"); ax.set_ylabel("genomes")
    ax.set_title("Genomes track cases")
    out["r2_cases_vs_genomes"] = r2
    out["zones_with_genomes"] = int((y > 0).sum()); out["zones"] = int(len(y))

    # ---------- (c) the one measurable delay ----------
    lag = (m["date"] - m["onset"]).dt.days.dropna()
    lag = lag[(lag >= 0) & (lag <= 40)]
    ax = axes[2]
    ax.hist(lag, bins=np.arange(0, 41, 2), color=SEQ, edgecolor="black", linewidth=0.3,
            density=True)
    q1, q3 = lag.quantile(.25), lag.quantile(.75)
    ax.axvspan(q1, q3, color=ACC, alpha=0.12, zorder=0)
    ax.axvline(lag.median(), color=ACC, lw=1.4)
    ax.text(0.95, 0.60, f"IQR {q1:.0f}--{q3:.0f} d\n$n$={len(lag)}", transform=ax.transAxes,
            ha="right", fontsize=6.4, color="#495057")
    ax.text(lag.median() + 1.2, ax.get_ylim()[1] * 0.86, f"median {lag.median():.0f} d",
            fontsize=7, color=ACC)
    ax.set_xlabel("onset to sample collection (days)"); ax.set_ylabel("density")
    ax.set_title("Collection delay")
    out["onset_to_collection_median_days"] = float(lag.median())
    out["onset_to_collection_iqr"] = [float(lag.quantile(.25)), float(lag.quantile(.75))]
    out["n_lag"] = int(len(lag))

    label_axes(fig, axes)
    fig.savefig(FIG / "seqintegration.pdf", bbox_inches="tight", pad_inches=0.015); plt.close(fig)
    json.dump(out, open(OUT / "paper_numbers_seqint.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
