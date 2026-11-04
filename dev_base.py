import sys, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import *
for m in sys.argv[1].split(","):
    tot = {}
    for dn in ["WA2014", "DRC1820"]:
        z, meta = load(dn); ds = {k: z[k] for k in z.files}; st, b = FOLDS[dn]; vals = []
        for s in st:
            d = splits(ds["inc"], s, b)
            for sd in range(1 if m in ("Naive","MA3") else 2):
                r, _, _ = train_eval(m, ds, d, sd, None); vals.append(r["val_MAE"])
        tot[dn] = round(float(np.mean(vals)), 3)
    print(m, tot, flush=True)
