# Run logs

Console output captured while building the benchmark. They are the record of what was
actually run; none of the pipeline reads them.

| Log | What it records |
|---|---|
| `run.log`, `seeds5.log` | The full rolling-origin benchmark, the five-seed run that produced `results/results.jsonl` |
| `mast.log` | The MAST-Net baseline added later, 35 runs at five seeds |
| `ref.log` | Naive and MA3 re-run at five seeds to confirm their seed variance is exactly zero |
| `sens.log` | Hyper-parameter and graph-construction sensitivity sweep |
| `posthoc.log` | Post-hoc calibrated variant |
| `train.log`, `train2.log`, `ptrain.log` | Per-epoch training curves for all 17 learned models |
| `diag.log`, `extra.log` | Diagnostics and scale-free metrics |
| `dev_base.log`, `dev_tune*.log` | Early development and tuning, kept for provenance |
| `pa*.log`, `ps*.log`, `pv*.log`, `pr*.log`, `n*.log`, `gi*.log`, `gp*.log`, `ct*.log`, `tr3.log`, `sci.log`, `nx.log`, `an.log` | Figure and analysis scripts; the numbered variants are successive re-runs after a figure or metric changed |
| `paper_*_r.log` | Final regeneration passes for the paper figures |

Logs with a numeric suffix are later re-runs of the same script, so the highest number
is the one whose output is in the paper.
