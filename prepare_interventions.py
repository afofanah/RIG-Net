"""
prepare_interventions.py - weekly response-intensity indicators U[t, i, k] aligned to the incidence panels.
All indicators at week t use only reports dated <= end of week t, so a forecast issued at origin t
(inputs = weeks < t) only ever sees U[t-1] and earlier.

k=0 lab_share   : cumulative confirmed / (confirmed + probable)   -> testing / isolation coverage
k=1 surv_load   : log1p(suspected cases reported) - log1p(cases in the last 3 weeks) -> case-finding effort
k=2 maturity    : log1p(weeks since the region's first reported case)                 -> response deployment
Missing values are forward-filled; before a region's first case all three are set to 0.
These are PROXIES of intervention intensity, not direct intervention logs.
"""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
from prepare_data import WA_COORDS, norm, UP

ROOT = Path(__file__).parent; OUT = ROOT / "data"


def weekly_last(df, col, weeks):
    s = df.groupby("Date")[col].max().sort_index()
    return s.reindex(s.index.union(weeks)).ffill().reindex(weeks).values


def assemble(inc, conf, prob, susp, weeks):
    T, N = inc.shape; U = np.zeros((T, N, 3), np.float32)
    cum = np.cumsum(inc, 0); started = cum > 0
    first = np.where(started.any(0), started.argmax(0), T)
    rec3 = np.stack([inc[max(0, t - 2):t + 1].sum(0) for t in range(T)])
    with np.errstate(invalid="ignore", divide="ignore"):
        lab = np.where((conf + prob) > 0, conf / (conf + prob), np.nan)
    lab = pd.DataFrame(lab).ffill().fillna(0.5).values
    surv = np.log1p(np.nan_to_num(pd.DataFrame(susp).ffill().values, nan=0)) - np.log1p(rec3)
    mat = np.log1p(np.clip(np.arange(T)[:, None] - first[None, :], 0, None))
    U[..., 0], U[..., 1], U[..., 2] = lab, surv, mat
    U[~started] = 0
    return U


def wa():
    meta = json.load(open(OUT / "WA2014.json")); weeks = pd.to_datetime(meta["weeks"])
    inc = np.load(OUT / "WA2014.npz")["inc"]
    d = pd.read_excel(UP / "data-ebola-public.xlsx", sheet_name="ROWCA Ebola All Sec Review")
    d = d[d.Country.isin(["Guinea", "Liberia", "Sierra Leone"])].copy()
    d["cat"] = d.Category.astype(str).str.strip().str.lower(); d["v"] = pd.to_numeric(d.Value, errors="coerce")
    iso = {"Guinea": "GIN", "Liberia": "LBR", "Sierra Leone": "SLE"}
    key = {(c, norm(n)): f"{c}:{n}" for (c, n) in WA_COORDS}; key[("SLE", "port")] = "SLE:Port Loko"
    d["node"] = [key.get((iso[c], norm(l))) for c, l in zip(d.Country, d.Localite)]
    d = d.dropna(subset=["node", "v"])
    T, N = inc.shape; conf, prob, susp = (np.full((T, N), np.nan) for _ in range(3))
    for j, n in enumerate(meta["nodes"]):
        g = d[d.node == n]
        for arr, cat in ((conf, "confirmed cases"), (prob, "probable cases"), (susp, "suspected cases")):
            gg = g[g.cat == cat].rename(columns={"v": "x"})
            if len(gg): arr[:, j] = weekly_last(gg, "x", weeks)
    return assemble(inc, conf, prob, susp, weeks)


def drc():
    meta = json.load(open(OUT / "DRC1820.json")); weeks = pd.to_datetime(meta["weeks"])
    inc = np.load(OUT / "DRC1820.npz")["inc"]
    d = pd.read_csv(glob.glob(str(UP / "*By-Health-Zone.csv"))[0], skiprows=[1])
    d["Date"] = pd.to_datetime(d.report_date)
    d["hz"] = d.health_zone.replace({"Kayina": "Kayna", "Rwampara (Bunia)": "Rwampara"})
    for c in ["confirmed_cases", "probable_cases", "total_suspected_cases"]: d[c] = pd.to_numeric(d[c], errors="coerce")
    T, N = inc.shape; conf, prob, susp = (np.full((T, N), np.nan) for _ in range(3))
    for j, n in enumerate(meta["nodes"]):
        g = d[d.hz == n]
        conf[:, j] = weekly_last(g, "confirmed_cases", weeks); prob[:, j] = weekly_last(g, "probable_cases", weeks)
        susp[:, j] = weekly_last(g, "total_suspected_cases", weeks)
    return assemble(inc, conf, prob, susp, weeks)


if __name__ == "__main__":
    for name, fn in (("WA2014", wa), ("DRC1820", drc)):
        U = fn(); np.save(OUT / f"{name}_interventions.npy", U)
        act = U[..., 2] > 0
        print(name, U.shape, "active region-weeks", int(act.sum()),
              "means(active) lab/surv/mat", np.round([U[..., k][act].mean() for k in range(3)], 3),
              "sd", np.round([U[..., k][act].std() for k in range(3)], 3))
