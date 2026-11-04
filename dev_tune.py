"""Design iteration for RIG-Net using VALIDATION MAE only (test blocks never inspected)."""
import sys, json, itertools, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import *
variants = json.loads(sys.argv[1]); seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
for name, v in variants.items():
    tot = {}
    for dn in ["WA2014", "DRC1820"]:
        z, meta = load(dn); ds = {k: z[k] for k in z.files}; st, b = FOLDS[dn]; vals = []
        for s in st:
            d = splits(ds["inc"], s, b)
            for sd in range(seeds):
                r, _, _ = train_eval(v.pop("_model", "RIG-Net") if "_model" in v else "RIG-Net", ds, d, sd, v); vals.append(r["val_MAE"])
        tot[dn] = round(float(np.mean(vals)), 3)
    print(name, tot, flush=True)
