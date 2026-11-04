"""
prepare_data.py
Builds weekly node-level incidence panels + static graphs for:
  WA2014  : Guinea + Liberia + Sierra Leone (tri-country cross-border graph), OCHA ROWCA, Mar-2014..Mar-2015
  DRC1820 : North Kivu / Ituri / South Kivu health zones, DRC MoH, Aug-2018..Jun-2020
  GIN2021 : Guinea Nzerekore 2021 re-emergence (zero-shot stress test only)

Coordinates are approximate seat-town coordinates (no shapefile was provided; the
process_shapefile.py helper expects a config + shapefile that were not uploaded).
"""
import json, re, unicodedata
from pathlib import Path
from paths import DATA, UPLOADS
import numpy as np
import pandas as pd

UP = UPLOADS
OUT = DATA; OUT.mkdir(parents=True, exist_ok=True)

def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"\b(county)\b", "", s.lower())
    return re.sub(r"[^a-z]", "", s)

# ---------------------------------------------------------------- coordinates
WA_COORDS = {  # (country, display name): (lat, lon) of administrative seat / centroid
 # Guinea prefectures
 ("GIN","Conakry"):(9.54,-13.68),("GIN","Gueckedou"):(8.56,-10.13),("GIN","Kissidougou"):(9.18,-10.10),
 ("GIN","Macenta"):(8.54,-9.47),("GIN","Telimele"):(10.90,-13.04),("GIN","Boffa"):(10.18,-14.04),
 ("GIN","Kouroussa"):(10.65,-9.88),("GIN","Dabola"):(10.75,-11.11),("GIN","Dubreka"):(9.79,-13.52),
 ("GIN","Nzerekore"):(7.75,-8.82),("GIN","Yomou"):(7.57,-9.25),("GIN","Dinguiraye"):(11.30,-10.72),
 ("GIN","Pita"):(11.06,-12.40),("GIN","Siguiri"):(11.42,-9.17),("GIN","Forecariah"):(9.43,-13.09),
 ("GIN","Kerouane"):(9.27,-9.01),("GIN","Coyah"):(9.71,-13.39),("GIN","Dalaba"):(10.69,-12.25),
 ("GIN","Kindia"):(10.06,-12.87),("GIN","Beyla"):(8.69,-8.64),("GIN","Lola"):(7.80,-8.53),
 ("GIN","Boke"):(10.94,-14.30),("GIN","Faranah"):(10.04,-10.74),("GIN","Mamou"):(10.38,-12.09),
 ("GIN","Kankan"):(10.39,-9.31),("GIN","Fria"):(10.37,-13.58),("GIN","Labe"):(11.32,-12.28),
 ("GIN","Mali"):(12.08,-12.30),("GIN","Tougue"):(11.45,-11.67),("GIN","Gaoual"):(11.75,-13.20),
 ("GIN","Koundara"):(12.48,-13.30),("GIN","Lelouma"):(11.42,-12.53),
 # Liberia counties
 ("LBR","Grand Cape Mount"):(7.05,-11.07),("LBR","Lofa"):(8.10,-9.70),("LBR","Margibi"):(6.52,-10.30),
 ("LBR","Nimba"):(6.85,-8.75),("LBR","Montserrado"):(6.55,-10.60),("LBR","RiverCess"):(5.90,-9.45),
 ("LBR","Bong"):(6.83,-9.37),("LBR","River Gee"):(5.26,-7.87),("LBR","Sinoe"):(5.50,-8.80),
 ("LBR","Grand Gedeh"):(5.92,-8.22),("LBR","Bomi"):(6.76,-10.85),("LBR","Grand Bassa"):(6.23,-9.81),
 ("LBR","Gbarpolu"):(7.50,-10.10),("LBR","Maryland"):(4.70,-7.70),("LBR","Grand Kru"):(4.76,-8.22),
 # Sierra Leone districts
 ("SLE","Kailahun"):(8.28,-10.57),("SLE","Kenema"):(7.88,-11.19),("SLE","Bo"):(7.96,-11.74),
 ("SLE","Moyamba"):(8.16,-12.43),("SLE","Bombali"):(9.20,-12.20),("SLE","Kambia"):(9.12,-12.92),
 ("SLE","Kono"):(8.64,-10.97),("SLE","Koinadugu"):(9.59,-11.55),("SLE","Bonthe"):(7.50,-12.30),
 ("SLE","Pujehun"):(7.35,-11.72),("SLE","Tonkolili"):(8.72,-11.95),("SLE","Western Area Rural"):(8.34,-13.07),
 ("SLE","Western Area Urban"):(8.48,-13.23),("SLE","Port Loko"):(8.77,-12.79),
}
DRC_COORDS = {
 "Beni":(0.49,29.47),"Katwa":(0.17,29.32),"Mabalako":(0.55,29.30),"Mandima":(1.05,29.10),
 "Butembo":(0.14,29.29),"Kalunguta":(0.30,29.35),"Vuhovi":(0.23,29.42),"Mambasa":(1.36,29.05),
 "Musienene":(0.05,29.27),"Komanda":(1.36,29.77),"Oicha":(0.70,29.52),"Masereka":(-0.13,29.33),
 "Lubero":(-0.16,29.24),"Mutwanga":(0.33,29.75),"Kyondo":(0.02,29.40),"Kayna":(-0.60,29.15),
 "Biena":(0.58,29.05),"Mangurujipa":(-0.40,28.95),"Rwampara":(1.53,30.20),"Alimbongo":(-0.37,29.17),
 "Mwenga":(-3.04,28.43),"Lolwa":(1.40,29.55),"Bunia":(1.56,30.25),"Nyiragongo":(-1.52,29.25),
 "Nyankunde":(1.44,30.04),"Tchomia":(1.47,30.51),"Ariwara":(3.14,30.70),"Pinga":(-0.98,28.68),
 "Goma":(-1.68,29.22),
}
GIN21_COORDS = {"Nzerekore":(7.75,-8.82),"Gouecke":(7.95,-8.80),"Soulouta":(7.70,-8.95),"Lola":(7.80,-8.53)}

def haversine(a, b):
    a, b = np.radians(a), np.radians(b)
    dlat = b[:, None, 0] - a[None, :, 0]; dlon = b[:, None, 1] - a[None, :, 1]
    h = np.sin(dlat/2)**2 + np.cos(a[None, :, 0])*np.cos(b[:, None, 0])*np.sin(dlon/2)**2
    return 6371*2*np.arcsin(np.sqrt(h))

def geo_graph(coords, sigma_km, k=6):
    """Gaussian distance kernel, kNN-sparsified, symmetric, zero diagonal."""
    D = haversine(coords, coords)
    W = np.exp(-(D/sigma_km)**2)
    np.fill_diagonal(W, 0)
    keep = np.zeros_like(W, bool)
    idx = np.argsort(D, 1)[:, 1:k+1]
    for i in range(len(W)): keep[i, idx[i]] = True
    W = np.where(keep | keep.T, W, 0)
    return W, D

def clean_cumulative(series, win=5):
    """Irregular cumulative reports -> robust monotone cumulative.
    1) centred rolling median over consecutive reports removes isolated spikes/typos
       and transient definition switches; 2) running max enforces monotonicity."""
    s = series.sort_index().astype(float)
    s = s.rolling(win, center=True, min_periods=1).median()
    return s.cummax()

def to_weekly_incidence(s, weeks, start_zero=True):
    """Linear interpolation of cumulative counts on a daily grid (spreads reporting
    gaps over the gap instead of dumping them into one week), then weekly differences."""
    days = pd.date_range(weeks[0] - pd.Timedelta(days=7), weeks[-1])
    s = s[~s.index.duplicated()]
    s = s.reindex(s.index.union(days)).interpolate("time", limit_area="inside")
    s = s.ffill().fillna(0 if start_zero else np.nan)
    c = s.reindex([weeks[0] - pd.Timedelta(days=7)] + list(weeks)).values
    c = np.maximum.accumulate(c)
    return np.diff(c)

# ---------------------------------------------------------------- WA2014
def build_wa():
    d = pd.read_excel(UP/"data-ebola-public.xlsx", sheet_name="ROWCA Ebola All Sec Review")
    d = d[d.Country.isin(["Guinea", "Liberia", "Sierra Leone"])].copy()
    d["cat"] = d.Category.astype(str).str.strip().str.lower()
    d = d[d.cat == "cases"]                       # cumulative confirmed+probable+suspected
    d["v"] = pd.to_numeric(d.Value, errors="coerce")
    d = d.dropna(subset=["v"]); d = d[d.Date >= "2014-03-01"]
    iso = {"Guinea": "GIN", "Liberia": "LBR", "Sierra Leone": "SLE"}
    key = {(c, norm(n)): (c, n) for (c, n) in WA_COORDS}
    alias = {("SLE", "port"): ("SLE", "Port Loko"), ("GIN", "guekedou"): ("GIN", "Gueckedou")}
    d["node"] = [key.get((iso[c], norm(l)), alias.get((iso[c], norm(l)))) for c, l in zip(d.Country, d.Localite)]
    d = d.dropna(subset=["node"])
    # common window in which all three countries publish sub-national series
    weeks = pd.date_range("2014-06-01", "2015-02-08", freq="W-SUN")
    nodes = sorted(d.node.unique(), key=lambda x: (x[0], x[1]))
    inc = np.zeros((len(weeks), len(nodes)))
    for j, n in enumerate(nodes):
        s = clean_cumulative(d[d.node == n].groupby("Date").v.max())
        inc[:, j] = to_weekly_incidence(s, weeks)
    return weeks, nodes, inc

# ---------------------------------------------------------------- DRC 2018-2020
def build_drc():
    d = pd.read_csv(next(UP.glob("*MOH-By-Health-Zone.csv")), skiprows=[1])
    d["report_date"] = pd.to_datetime(d.report_date)
    d = d[d.province.isin(["North Kivu", "Ituri", "South Kivu"])].copy()
    d["hz"] = d.health_zone.replace({"Kayina": "Kayna", "Rwampara (Bunia)": "Rwampara"})
    d["v"] = pd.to_numeric(d.total_cases, errors="coerce")
    d = d.dropna(subset=["v"])
    tot = d.groupby("hz").v.max(); nodes = sorted(tot[tot > 0].index)
    nodes = [n for n in nodes if n in DRC_COORDS]
    weeks = pd.date_range("2018-08-05", "2020-06-21", freq="W-SUN")
    inc = np.zeros((len(weeks), len(nodes)))
    for j, n in enumerate(nodes):
        s = clean_cumulative(d[d.hz == n].groupby("report_date").v.max())
        inc[:, j] = to_weekly_incidence(s, weeks)
    # Rwampara (Bunia) duplication guard: if merged sum double-counts, use max instead
    pop = pd.read_csv(UP/"worldpop__pop_count__static.csv").set_index("nom").pop_count
    dens = pd.read_csv(UP/"worldpop__pop_density__static.csv").set_index("nom").pop_density
    hs = pd.read_csv(UP/"healthsites_io__healthsite_count__static.csv").set_index("nom").healthsite_count
    stat = pd.DataFrame({"pop": pop.reindex(nodes), "dens": dens.reindex(nodes), "hs": hs.reindex(nodes)})
    stat = stat.fillna(stat.median())   # Mangurujipa missing in WorldPop table -> median imputation
    return weeks, nodes, inc, stat

# ---------------------------------------------------------------- GIN2021
def build_gin21():
    g = pd.read_excel(UP/"guinee-rapport-de-situation-sur-lepidemie-de-la-maladie-a-virus-ebola.xlsx", sheet_name="Cas")
    g = g[g.Indicateur.isin(["Cas Confirmés", "Cas Probables"])]
    g["loc"] = g["Localité"].map(lambda s: {"cudenzerekore": "Nzerekore", "nzerekore": "Nzerekore",
                                             "gouecke": "Gouecke", "soulouta": "Soulouta", "lola": "Lola"}.get(norm(s)))
    g = g.dropna(subset=["loc"])
    g = g.groupby(["loc", "Date des données", "Indicateur"]).Cumulatif.max().groupby(["loc", "Date des données"]).sum()
    nodes = sorted(GIN21_COORDS); weeks = pd.date_range("2021-01-31", "2021-06-20", freq="W-SUN")
    inc = np.zeros((len(weeks), len(nodes)))
    for j, n in enumerate(nodes):
        if n not in g.index.get_level_values(0): continue
        s = g.loc[n]; s.index = pd.to_datetime(s.index)
        inc[:, j] = to_weekly_incidence(clean_cumulative(s, win=1), weeks)
    return weeks, nodes, inc

def save(name, weeks, labels, coords, inc, sigma, static=None):
    W, D = geo_graph(np.array(coords), sigma)
    np.savez(OUT/f"{name}.npz", inc=inc.astype(np.float32), W=W.astype(np.float32), D=D.astype(np.float32),
             coords=np.array(coords, np.float32),
             static=(static if static is not None else np.zeros((len(labels), 0))).astype(np.float32))
    meta = {"name": name, "weeks": [str(w.date()) for w in weeks], "nodes": labels,
            "T": int(inc.shape[0]), "N": int(inc.shape[1]), "total_cases": float(inc.sum()),
            "zero_frac": float((inc == 0).mean())}
    json.dump(meta, open(OUT/f"{name}.json", "w"), indent=1)
    pd.DataFrame(inc, index=[str(w.date()) for w in weeks], columns=labels).to_csv(OUT/f"{name}_weekly_incidence.csv")
    print(name, {k: v for k, v in meta.items() if k not in ("weeks", "nodes")})

def rebuild_graphs():
    """Geographic kernels at other (sigma, k) settings, for the graph-construction sensitivity test."""
    import itertools
    for name, sig0 in (("WA2014", 150), ("DRC1820", 40)):
        z = np.load(OUT / f"{name}.npz"); coords = z["coords"]
        for sf, k in itertools.product((0.5, 2.0), (3, 6, 12)):
            if sf == 1.0 and k == 6: continue
            W, _ = geo_graph(coords, sig0 * sf, k=k)
            np.savez(OUT / f"{name}_geo_s{sf}_k{k}.npz", W=W.astype(np.float32))


if __name__ == "__main__":
    w, n, inc = build_wa()
    save("WA2014", w, [f"{c}:{x}" for c, x in n], [WA_COORDS[x] for x in n], inc, sigma=150)
    w, n, inc, st = build_drc()
    static = np.column_stack([np.log1p(st["pop"]), np.log1p(st["dens"]), np.log1p(st["hs"])])
    save("DRC1820", w, n, [DRC_COORDS[x] for x in n], inc, sigma=40, static=static)
    w, n, inc = build_gin21()
    save("GIN2021", w, n, [GIN21_COORDS[x] for x in n], inc, sigma=40)
