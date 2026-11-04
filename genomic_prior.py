"""
genomic_prior.py - leakage-safe genomic transmission-linkage graph for WA2014.

Data: 1,610 EBOV Makona genomes with district and collection date
(Dudas et al., Nature 2017; github.com/ebov/space-time, Data/).

For a cut-off date c (the end of a fold's training period) only genomes collected before c
are used. Two genomes are 'linked' when their pairwise SNP distance (over sites called in
both) is <= tau. The genomic affinity between districts i != j is
    G_ij = (# linked genome pairs between i and j) / sqrt(n_i * n_j)
which is symmetric, and 0 for districts without sequenced genomes before c.
"""
import json, re, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent; EXT = ROOT / "external"


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


ALIAS = {"westernarea": "westernareaurban", "westernurban": "westernareaurban", "westernrural": "westernarearural", "guekedou": "gueckedou",
         "grandcapemount": "grandcapemount", "rivercess": "rivercess", "nzerekore": "nzerekore",
         "forecariah": "forecariah", "telimele": "telimele"}


def read_fasta(path):
    seqs, names, cur = [], [], []
    for line in open(path, encoding="latin-1"):
        line = line.strip()
        if line.startswith(">"):
            if cur: seqs.append("".join(cur)); cur = []
            names.append(line[1:])
        else: cur.append(line.upper())
    if cur: seqs.append("".join(cur))
    L = min(len(s) for s in seqs)
    return names, np.frombuffer("".join(s[:L] for s in seqs).encode(), dtype=np.uint8).reshape(len(seqs), L)


def load_drc(nodes):
    """852 Nord Kivu genomes (Kinganda-Lusamaki et al., Nat Med 2021; github.com/inrb-drc/ebola-nord-kivu),
    aligned to BEN214 with MAFFT --keeplength."""
    names, A = read_fasta(EXT / "drc_aligned.fasta")
    md = pd.read_csv(EXT / "drc_metadata.tsv", sep="\t").set_index("strain")
    hz = md.health_zone.reindex(names).replace({"Kayina": "Kayna", "Manguredjipa": "Mangurujipa", "Nyakunde": "Nyankunde"}).values
    date = pd.to_datetime(md.date.reindex(names).values, errors="coerce")
    idx = {norm(n): j for j, n in enumerate(nodes)}
    node = np.array([idx.get(norm(h), -1) if isinstance(h, str) else -1 for h in hz])
    return A, pd.DataFrame({"node": node, "date": date, "location": hz})


def load_wa(nodes):
    A, meta = load_genomes()
    key = {}
    for j, n in enumerate(nodes):
        c, name = n.split(":"); key[(c, norm(name))] = j
    meta["node"] = [key.get((c, ALIAS.get(norm(l), norm(l))), -1) for c, l in zip(meta.country, meta.location)]
    return A, meta


def load_genomes():
    seqs, names = [], []
    cur = []
    for line in open(EXT / "Makona_1610_genomes_2016-06-23.fasta", encoding="latin-1"):
        line = line.strip()
        if line.startswith(">"):
            if cur: seqs.append("".join(cur)); cur = []
            names.append(line[1:])
        else: cur.append(line.upper())
    if cur: seqs.append("".join(cur))
    L = min(len(s) for s in seqs)
    A = np.frombuffer("".join(s[:L] for s in seqs).encode(), dtype=np.uint8).reshape(len(seqs), L)
    meta = pd.DataFrame([n.split("|") for n in names], columns=["virus", "id", "acc", "country", "location", "date"])
    meta["date"] = pd.to_datetime(meta.date, errors="coerce")
    return A, meta


def snp_distance(A):
    bases = [ord(c) for c in "ACGT"]
    onehot = np.stack([(A == b) for b in bases], 0)                 # [4, n, L]
    called = onehot.any(0)
    var = np.zeros(A.shape[1], bool)
    for b in range(4):
        frac = onehot[b].sum(0)
        var |= (frac > 0) & (frac < called.sum(0))
    X = onehot[:, :, var].astype(np.float32); C = called[:, var].astype(np.float32)
    same = sum(X[b] @ X[b].T for b in range(4)); both = C @ C.T
    return both - same, int(var.sum())


def build(nodes, cutoffs, loader, tau=2, frac=1.0, seed=0):
    """cutoffs: list of pd.Timestamp. Returns {cutoff: G (N x N)} using genomes collected before each cutoff."""
    A, meta = loader(nodes)
    if frac < 1.0:
        rng = np.random.default_rng(seed)
        keep = rng.random(len(meta)) < frac
        meta = meta.copy(); meta.loc[~keep, "node"] = -1
    D, nvar = snp_distance(A)
    stats = {"genomes": len(meta), "variable_sites": nvar, "mapped_to_nodes": int((meta.node >= 0).sum()),
             "unmapped_locations": sorted(set(meta.location[meta.node < 0]))[:30]}
    out = {}
    for c in cutoffs:
        ok = ((meta.node >= 0) & (meta.date < c)).values
        idx = np.where(ok)[0]; nd = meta.node.values[idx]
        link = (D[np.ix_(idx, idx)] <= tau)
        N = len(nodes); G = np.zeros((N, N)); cnt = np.bincount(nd, minlength=N).astype(float)
        np.add.at(G, (nd[:, None].repeat(len(idx), 1), nd[None, :].repeat(len(idx), 0)), link)
        np.fill_diagonal(G, 0)
        G = G / np.sqrt(np.outer(cnt, cnt) + 1e-9)
        out[str(c.date())] = {"G": G, "n_genomes": int(ok.sum()), "n_nodes_with_genomes": int((cnt > 0).sum())}
    return out, stats


if __name__ == "__main__":
    from run_experiments import FOLDS, splits
    for dn, loader in (("WA2014", load_wa), ("DRC1820", load_drc)):
        meta_ds = json.load(open(ROOT / "data" / f"{dn}.json")); z = np.load(ROOT / "data" / f"{dn}.npz")
        weeks = pd.to_datetime(meta_ds["weeks"]); cut = []
        for s in FOLDS[dn][0]:
            v0 = splits(z["inc"], s, FOLDS[dn][1])[3][0]; cut.append(weeks[v0])
        for fr in (0.25, 0.5, 0.75):
            rv, _ = build(meta_ds["nodes"], cut, loader, frac=fr)
            np.savez(ROOT / "data" / f"{dn}_genomic_f{int(fr*100)}.npz", **{f"fold{f}": v["G"] for f, v in enumerate(rv.values())})
        for tv in (1, 2, 4):
            rv, stats = build(meta_ds["nodes"], cut, loader, tau=tv)
            if tv != 2:
                np.savez(ROOT / "data" / f"{dn}_genomic_tau{tv}.npz", **{f"fold{f}": v["G"] for f, v in enumerate(rv.values())})
        res, stats = build(meta_ds["nodes"], cut, loader)
        print(dn, json.dumps(stats))
        np.savez(ROOT / "data" / f"{dn}_genomic.npz", **{f"fold{f}": v["G"] for f, v in enumerate(res.values())})
        summ = {}
        for f, (k, v) in enumerate(res.items()):
            G = v["G"]; iu = np.dstack(np.unravel_index(np.argsort(-G.ravel())[:10:2], G.shape))[0]
            summ[k] = {"genomes_before_cutoff": v["n_genomes"], "regions_with_genomes": v["n_nodes_with_genomes"],
                       "edges": int((G > 0).sum() // 2),
                       "top_links": [(meta_ds["nodes"][i], meta_ds["nodes"][j], round(float(G[i, j]), 2)) for i, j in iu]}
            print(" ", k, summ[k])
        json.dump({"stats": stats, "folds": summ}, open(ROOT / "data" / f"{dn}_genomic_summary.json", "w"), indent=1)
