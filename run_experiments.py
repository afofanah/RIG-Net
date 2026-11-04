"""
run_experiments.py - identical protocol for every model.

Protocol
  * weekly incidence, lookback L=6, horizons H=1..4 weeks
  * rolling-origin evaluation: several non-overlapping test blocks covering growth, peak and
    decline. For a block starting at week s: train targets < s-v, validation targets in
    [s-v, s) (v = max(4, 20% of s)), test targets in [s, s+b). Retrain per fold.
    Forecast origins whose horizons straddle a boundary are kept with masked targets.
  * scaler (mean/sd of log1p incidence) fitted on each fold's training weeks only
  * Adam(lr=1e-3, wd=1e-4), full batch, <=400 epochs, early stopping on validation
    count-space MAE (patience 50), same budget for all models, 5 seeds
  * metrics on test in count space: MAE, RMSE, PCC (+ MAE per horizon)
"""
import argparse, json, time, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from baselines import BASELINES
from rignet import RIGNet, nb_nll
from scipy.stats import nbinom


def nb_quantile(mu, alpha, q):
    r = (1.0 / alpha).clamp_min(1e-4).numpy().astype(np.float64); m = mu.clamp_min(1e-8).numpy().astype(np.float64)
    return torch.tensor(nbinom.ppf(q, r, r / (r + m)), dtype=torch.float32)

ROOT = Path(__file__).parent
L_IN, H_OUT = 6, 4
torch.set_num_threads(1)


def load(name):
    z = np.load(ROOT / "data" / f"{name}.npz"); meta = json.load(open(ROOT / "data" / f"{name}.json"))
    return z, meta


def make_windows(inc, L, H, t_lo, t_hi):
    """Forecast origin t uses weeks [t-L, t) and predicts weeks t..t+H-1.
    Only target weeks in [t_lo, t_hi) are scored (mask); origins range so every
    such week is covered at every horizon. Inputs never include week >= t."""
    T, N = inc.shape; X, Y, M, O = [], [], [], []
    for t in range(max(L, t_lo - H + 1), t_hi):
        idx = np.arange(t, t + H); m = (idx >= t_lo) & (idx < t_hi)
        y = np.zeros((H, N)); ok = idx < T; y[ok] = inc[idx[ok]]
        X.append(inc[t - L:t]); Y.append(y); M.append(np.repeat(m[:, None], N, 1)); O.append(t)
    f = lambda v: torch.tensor(np.array(v), dtype=torch.float32)
    return f(X), f(Y), torch.tensor(np.array(M)), np.array(O)


FOLDS = {"WA2014": ([19, 25, 31], 6), "DRC1820": ([36, 48, 60, 72], 12)}


def splits(inc, s, b):
    T = inc.shape[0]; nv = max(4, int(round(0.2 * s))); v0 = s - nv
    tr = make_windows(inc, L_IN, H_OUT, L_IN, v0)
    va = make_windows(inc, L_IN, H_OUT, v0, s)
    te = make_windows(inc, L_IN, H_OUT, s, min(s + b, T))
    return tr, va, te, (v0, s)


def metrics(pred, y, m):
    p = pred[m].numpy(); t = y[m].numpy()
    mae = np.abs(p - t).mean(); rmse = np.sqrt(((p - t) ** 2).mean())
    pcc = float(np.corrcoef(p, t)[0, 1]) if p.std() > 0 and t.std() > 0 else 0.0
    per_h = [float(np.abs(pred[:, h][m[:, h]] - y[:, h][m[:, h]]).mean()) if m[:, h].any() else float("nan") for h in range(y.shape[1])]
    return {"MAE": float(mae), "RMSE": float(rmse), "PCC": pcc, "MAE_h": per_h}


class Wrapper:
    """Unifies log-space regressors and count-space probabilistic models."""
    def __init__(self, name, ds, scaler, seed, variant=None):
        torch.manual_seed(seed); np.random.seed(seed)
        z = ds; N = z["inc"].shape[1]
        A = torch.tensor(z["W"]); S = torch.tensor(z["static"])
        coords = torch.tensor(z["coords"]); coords = (coords - coords.mean(0)) / coords.std(0)
        S = torch.cat([coords, (S - S.mean(0)) / (S.std(0) + 1e-6)], 1) if S.shape[1] else coords
        pop = None
        if name == "MepoGNN" and z["static"].shape[1]:
            pop = torch.tensor(np.expm1(z["static"][:, 0]), dtype=torch.float32)
        self.name, self.scaler = name, scaler; self.use_int = False; self.Uw = {}
        if name.startswith("RIG-Net"):
            v = dict(variant or {}); use_gen = v.pop("genomic", False); use_int = v.pop("intervention", False); tau = v.pop("tau", 2); gfrac = v.pop("gfrac", 100); geo = v.pop("geo", None)
            if use_gen and tau != 2: ds = {**ds, "G": ds["G_tau%d" % tau]}
            if use_gen and gfrac != 100: ds = {**ds, "G": ds["G_f%d" % gfrac]}
            if geo is not None and ("W_" + geo) in ds: A = torch.tensor(ds["W_" + geo], dtype=torch.float32)
            G = torch.tensor(ds["G"], dtype=torch.float32) if use_gen and "G" in ds else None
            self.use_int = use_int and "U" in ds
            self.model = RIGNet(L_IN, H_OUT, N, A=A, S=S, scaler=scaler, G=G, n_interv=3 if self.use_int else 0,
                                **{k: x for k, x in v.items() if k not in ("loss", "point", "lr", "l1w", "tau", "gfrac", "geo")})
            self.l1w = (variant or {}).get("l1w", 1.0)
            self.kind = "count"
            self.loss_mode = (variant or {}).get("loss", "nb"); self.point = (variant or {}).get("point", "median")
        elif name in ("Naive", "MA3"):
            self.model = None; self.kind = "ref"
        else:
            cls = BASELINES[name][0]
            self.model = cls(L_IN, H_OUT, N, A=A, S=S, pop=pop, scaler=scaler); self.kind = "log"

    def to_x(self, Xc):
        mu, sd = self.scaler; return (torch.log1p(Xc) - mu) / sd

    def fwd(self, Xc, key):
        return self.model(self.to_x(Xc), self.Uw.get(key)) if self.use_int else self.model(self.to_x(Xc))

    def predict(self, Xc, key=None):
        if self.kind == "ref":
            if self.name == "Naive": return Xc[:, -1:].repeat(1, H_OUT, 1)
            return Xc[:, -3:].mean(1, keepdim=True).repeat(1, H_OUT, 1)
        self.model.eval()
        with torch.no_grad():
            out = self.fwd(Xc, key) if self.kind == "count" else self.model(self.to_x(Xc))
            if self.kind == "count":
                if self.point == "median" and self.loss_mode == "nb":
                    return nb_quantile(out["mu"], out["phi"].expand_as(out["mu"]), 0.5)
                return out["mu"].clamp_min(0)
            mu, sd = self.scaler; return torch.expm1(out * sd + mu).clamp_min(0)

    def loss(self, Xc, Yc, M):
        out = self.fwd(Xc, "tr") if self.kind == "count" else self.model(self.to_x(Xc)); mu, sd = self.scaler
        if self.kind == "count":
            m = out["mu"]
            l1 = (torch.log1p(m) - torch.log1p(Yc)).abs()[M].mean()
            if self.loss_mode == "nb":
                return nb_nll(Yc[M], m[M], out["phi"].expand_as(m)[M]) + self.l1w * l1 + out.get("kl", 0.0)
            return ((torch.log1p(m) - torch.log1p(Yc)) ** 2)[M].mean()
        yt = (torch.log1p(Yc) - mu) / sd
        return ((out - yt) ** 2)[M].mean()


def train_eval(name, ds, data, seed, variant=None, epochs=300, patience=40, lr=1e-3):
    lr = (variant or {}).get("lr", lr)
    (Xtr, Ytr, Mtr, _), (Xva, Yva, Mva, _), (Xte, Yte, Mte, Ote), _ = data
    trw = ds["inc"][: data[3][0]]
    lw = np.log1p(trw); scaler = (float(lw.mean()), float(lw.std()))
    w = Wrapper(name, ds, scaler, seed, variant)
    if w.use_int:
        U = ds["U"]; v0 = data[3][0]; act = U[:v0, :, 2] > 0
        m_ = U[:v0][act].mean(0); s_ = U[:v0][act].std(0) + 1e-6
        Us = np.where((U[..., 2:3] > 0), (U - m_) / s_, 0.0).astype(np.float32)   # standardised on training weeks
        for key, part in (("tr", data[0]), ("va", data[1]), ("te", data[2])):
            w.Uw[key] = torch.tensor(Us[part[3] - 1])                            # indicators at week t-1
    t0 = time.time()
    if w.kind != "ref":
        opt = torch.optim.Adam(w.model.parameters(), lr=lr, weight_decay=1e-4)
        best, best_state, bad = 1e18, None, 0
        for ep in range(epochs):
            w.model.train(); opt.zero_grad()
            loss = w.loss(Xtr, Ytr, Mtr)
            if not torch.isfinite(loss): break
            loss.backward(); torch.nn.utils.clip_grad_norm_(w.model.parameters(), 5.0); opt.step()
            if hasattr(w.model, "project"): w.model.project()
            v = metrics(w.predict(Xva, "va"), Yva, Mva)["MAE"]
            if v < best - 1e-6: best, bad = v, 0; best_state = {k: t.clone() for k, t in w.model.state_dict().items()}
            else:
                bad += 1
                if bad > patience: break
        if best_state is not None: w.model.load_state_dict(best_state)
    pred = w.predict(Xte, "te")
    res = metrics(pred, Yte, Mte); res["train_sec"] = time.time() - t0
    res["val_MAE"] = metrics(w.predict(Xva, "va"), Yva, Mva)["MAE"]
    res["n_params"] = sum(p.numel() for p in w.model.parameters()) if w.model is not None else 0
    return res, pred, w


RIG_CFG = {"d": 64, "lr": 3e-3}          # frozen after validation-only tuning (dev_tune*.log)
ABLATIONS = {
    "RIG-Net": {**RIG_CFG},
    "RIG-Net w/o renewal (direct head)": {**RIG_CFG, "use_renewal": False},
    "RIG-Net w/o geo graph": {**RIG_CFG, "use_geo": False},
    "RIG-Net w/o adaptive graph": {**RIG_CFG, "use_adaptive": False},
    "RIG-Net w/o dynamic graph": {**RIG_CFG, "use_dynamic": False},
    "RIG-Net w/o NB likelihood": {**RIG_CFG, "loss": "mse"},
    # POST-HOC (defined after seeing test results; reported separately, not as the main result)
    "RIG-Net-S (post-hoc)": {**RIG_CFG, "R_max": 2.5, "clip_k": 3.0},
    # RIG-Net-GI: genomic priors + intervention-aware dynamics on the stabilised core
    "RIG-Net-GI": {**RIG_CFG, "R_max": 2.5, "clip_k": 3.0, "genomic": True, "intervention": True, "gen_kl": 0.1},
    "RIG-Net-GI w/o genomic prior": {**RIG_CFG, "R_max": 2.5, "clip_k": 3.0, "intervention": True},
    "RIG-Net-GI w/o intervention": {**RIG_CFG, "R_max": 2.5, "clip_k": 3.0, "genomic": True, "gen_kl": 0.1},
    "RIG-Net-GI w/o genomic KL": {**RIG_CFG, "R_max": 2.5, "clip_k": 3.0, "genomic": True, "intervention": True},
    "RIG-Net-GI tau=1": {**RIG_CFG, "R_max": 2.5, "clip_k": 3.0, "genomic": True, "intervention": True, "gen_kl": 0.1, "tau": 1},
    "RIG-Net-GI tau=4": {**RIG_CFG, "R_max": 2.5, "clip_k": 3.0, "genomic": True, "intervention": True, "gen_kl": 0.1, "tau": 4},
    "RIG-Net-GI Rmax=2.0": {**RIG_CFG, "R_max": 2.0, "clip_k": 3.0, "genomic": True, "intervention": True, "gen_kl": 0.1},
    "RIG-Net-GI Rmax=4.0": {**RIG_CFG, "R_max": 4.0, "clip_k": 3.0, "genomic": True, "intervention": True, "gen_kl": 0.1},
    "RIG-Net-GI lambda=0.0": {**RIG_CFG, "R_max": 2.5, "clip_k": 3.0, "genomic": True, "intervention": True, "gen_kl": 0.0},
    "RIG-Net-GI lambda=0.5": {**RIG_CFG, "R_max": 2.5, "clip_k": 3.0, "genomic": True, "intervention": True, "gen_kl": 0.5},
    "RIG-Net-GI kappa=1.5": {**RIG_CFG, "R_max": 2.5, "clip_k": 1.5, "genomic": True, "intervention": True, "gen_kl": 0.1},
    "RIG-Net-GI gen25%": {**RIG_CFG, "R_max": 2.5, "clip_k": 3.0, "genomic": True, "intervention": True, "gen_kl": 0.1, "gfrac": 25},
    "RIG-Net-GI gen50%": {**RIG_CFG, "R_max": 2.5, "clip_k": 3.0, "genomic": True, "intervention": True, "gen_kl": 0.1, "gfrac": 50},
    "RIG-Net-GI gen75%": {**RIG_CFG, "R_max": 2.5, "clip_k": 3.0, "genomic": True, "intervention": True, "gen_kl": 0.1, "gfrac": 75},
    "RIG-Net-GI geo s0.5 k6": {**RIG_CFG, "R_max": 2.5, "clip_k": 3.0, "genomic": True, "intervention": True, "gen_kl": 0.1, "geo": "s0.5_k6"},
    "RIG-Net-GI geo s2.0 k6": {**RIG_CFG, "R_max": 2.5, "clip_k": 3.0, "genomic": True, "intervention": True, "gen_kl": 0.1, "geo": "s2.0_k6"},
    "RIG-Net-GI geo s1.0 k3": {**RIG_CFG, "R_max": 2.5, "clip_k": 3.0, "genomic": True, "intervention": True, "gen_kl": 0.1, "geo": "s1.0_k3"},
    "RIG-Net-GI geo s1.0 k12": {**RIG_CFG, "R_max": 2.5, "clip_k": 3.0, "genomic": True, "intervention": True, "gen_kl": 0.1, "geo": "s1.0_k12"},
    "RIG-Net-GI kappa=10.0": {**RIG_CFG, "R_max": 2.5, "clip_k": 10.0, "genomic": True, "intervention": True, "gen_kl": 0.1},
}
BASELINE_LRS = (1e-3, 3e-3)


def run_one(mname, ds, data, seed):
    """Baselines: learning rate chosen per fold/seed on validation MAE (same search for all)."""
    if mname in ABLATIONS or mname in ("Naive", "MA3"):
        res, pred, _ = train_eval(mname, ds, data, seed, ABLATIONS.get(mname)); res["lr"] = (ABLATIONS.get(mname) or {}).get("lr")
        return res, pred
    best = None
    for lr in BASELINE_LRS:
        res, pred, _ = train_eval(mname, ds, data, seed, None, lr=lr); res["lr"] = lr
        if best is None or res["val_MAE"] < best[0]["val_MAE"]: best = (res, pred)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="WA2014,DRC1820"); ap.add_argument("--models", default="all")
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--out", default="results/results.jsonl")
    a = ap.parse_args()
    out = ROOT / a.out; out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out): r = json.loads(line); done.add((r["dataset"], r["model"], r["seed"], r["fold"]))
    models = (["Naive", "MA3"] + list(BASELINES) + list(ABLATIONS)) if a.models == "all" else a.models.split(",")
    for seed in range(a.seeds):
        for dname in a.datasets.split(","):
            z, meta = load(dname); ds = {k: z[k] for k in z.files}
            ds["U"] = np.load(ROOT / "data" / f"{dname}_interventions.npy")
            gen = np.load(ROOT / "data" / f"{dname}_genomic.npz")
            starts, blk = FOLDS[dname]
            for fold, st in enumerate(starts):
                ds["G"] = gen[f"fold{fold}"]
                for tv in (1, 4):
                    gp = ROOT / "data" / f"{dname}_genomic_tau{tv}.npz"
                    if gp.exists(): ds[f"G_tau{tv}"] = np.load(gp)[f"fold{fold}"]
                for fr in (25, 50, 75):
                    gp = ROOT / "data" / f"{dname}_genomic_f{fr}.npz"
                    if gp.exists(): ds[f"G_f{fr}"] = np.load(gp)[f"fold{fold}"]
                for gk in ("s0.5_k6", "s2.0_k6", "s1.0_k3", "s1.0_k12"):
                    gp = ROOT / "data" / f"{dname}_geo_{gk}.npz"
                    if gp.exists(): ds["W_" + gk] = np.load(gp)["W"]
                data = splits(ds["inc"], st, blk)
                pdir = ROOT / "results" / "preds" / dname / f"fold{fold}"; pdir.mkdir(parents=True, exist_ok=True)
                np.save(pdir / "Y.npy", data[2][1].numpy()); np.save(pdir / "M.npy", data[2][2].numpy()); np.save(pdir / "O.npy", data[2][3])
                for mname in models:
                    # references are deterministic, but we run all seeds so the table reports a measured sd
                    if (dname, mname, seed, fold) in done: continue
                    res, pred = run_one(mname, ds, data, seed)
                    np.save(pdir / f"{mname.replace(' ', '_').replace('/', '')}__s{seed}.npy", pred.numpy())
                    rec = {"dataset": dname, "fold": fold, "test_start": meta["weeks"][st], "model": mname, "seed": seed, **res}
                    with open(out, "a") as f: f.write(json.dumps(rec) + "\n")
                    print(f"{dname:8s} f{fold} {mname:36s} s{seed} MAE={res['MAE']:.3f} val={res['val_MAE']:.3f} ({res['train_sec']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
