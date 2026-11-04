# RIG-Net: Renewal–Importation Graph Network for sub-national Ebola forecasting

Benchmark of a renewal-equation graph neural network against 16 forecasting models
published 2020–2026, on weekly sub-national Ebola incidence from Guinea, Liberia,
Sierra Leone (2014–15) and the Democratic Republic of the Congo (2018–20).

## Environment
CPU only. Python 3.12, PyTorch 2.14 (conda-forge CPU build), numpy, pandas, scipy, matplotlib, openpyxl.

## Pipeline
```
python prepare_data.py      # raw uploads -> data/*.npz, *_weekly_incidence.csv
python dev_tune.py '<json>' # validation-only design iteration for RIG-Net (logs: dev_tune*.log)
python run_experiments.py --seeds 5   # full rolling-origin benchmark (1,446 runs) -> results/results.jsonl, results/preds/
python analyze.py           # tables, significance tests, figures -> results/
python diagnostics.py       # RIG-Net interpretability (R, importation share, graph gates)
```
`prepare_data.py` expects the original uploads in `/mnt/user-data/uploads`; edit `UP` to point elsewhere.

## Files
| File | Purpose |
|---|---|
| `prepare_data.py` | Cleaning (rolling-median outlier removal, monotone cumulative, gap interpolation), weekly panels, distance-kernel graphs |
| `baselines.py` | Compact re-implementations of the 16 baselines (core mechanisms; not official code). MAST-Net is reconstructed from its published description, the full text being unavailable |
| `rignet.py` | RIG-Net model and negative-binomial likelihood |
| `run_experiments.py` | Rolling-origin protocol, training loop, per-fold learning-rate search for baselines |
| `analyze.py`, `diagnostics.py` | Aggregation, Wilcoxon + Holm and Diebold–Mariano tests, figures |
| `genomic_prior.py` | SNP-distance transmission-linkage graphs, rebuilt per fold from genomes before the cut-off |
| `prepare_interventions.py` | Surveillance-derived response indicators (lab-confirmation share, surveillance load, response maturity) |
| `paper_results.py` | Pooled metrics, Diebold–Mariano tests, phase and horizon figures |
| `paper_analysis.py` | Validation-vs-test transfer, spatio-temporal error, R field, importation flows |
| `paper_science.py` | Spatial panels, genomic validation, theory checks, time-step predictions |
| `paper_validity.py` | Leakage timeline, EpiEstim comparison, calibration, critical-difference diagram, dose–response |
| `paper_pointpattern.py` | Ripley's L against a reshuffling null, spread periods, temporal decomposition |
| `paper_hotspot.py` | Voronoi zones and Getis-Ord hot/cold spots |
| `paper_sources.py` | External check against the WHO two-source data package |
| `paper_tracing.py` | Predicted onward-transmission networks and their lagged correlation |
| `paper_nexus.py`, `paper_country_ts.py`, `paper_training.py`, `paper_problem.py` | Health-access nexus, national forecasts, training dynamics, Fig. 1 |
| `logs/` | Console output from every run, with an index in `logs/README.md` |
| `data/` | Cleaned weekly incidence (CSV) and model-ready arrays |
| `results/` | `results.jsonl` (every run), `main_table.csv`, `significance.csv`, `per_horizon.csv`, `per_phase.csv`, figures |

## Protocol summary
* Lookback 6 weeks, horizons 1–4 weeks, errors in weekly case counts.
* Rolling origin: WA2014 test blocks start 2014-10-12, 2014-11-23, 2015-01-04 (6 weeks each);
  DRC1820 test blocks start 2019-04-14, 2019-07-07, 2019-09-29, 2019-12-22 (12 weeks each).
  Each fold: validation = the max(4, 20%) weeks before the test block; train = everything earlier.
* Five random seeds per model; metrics reported as mean ± s.d.
* Identical budget: Adam, full batch, ≤300 epochs, early stopping on validation MAE (patience 40).
  Baselines choose lr ∈ {1e-3, 3e-3} per fold/seed on validation; RIG-Net uses d=64, lr=3e-3, fixed
  beforehand from validation-only tuning.
* Variant selection for RIG-Net was pre-registered on validation scores (normalised by the naive
  forecast) and picked the full model.

## Known limitations
* Region coordinates are approximate seat-town locations (no shapefile was supplied).
* WA2014 contains a genuine Liberian reporting backlog (Montserrado, week of 2014-10-26).
* Baselines are re-implementations sized for small panels; official implementations and larger
  hyper-parameter searches might score differently.
* Guinea 2021 (26 cases in 4 localities) is too sparse for training or meaningful evaluation and is
  only provided as a cleaned series.

## Paths

All filesystem locations are defined once in `paths.py`:

```python
PROJECT = /Users/s5273738/RIG-Net     # project root
  ebola_gnn/   CODE      scripts, data/, results/, external/
  uploads/     UPLOADS   raw inputs (WHO package, shapefiles)
  paper/       PAPER     LaTeX source, figs/ written by the paper_*.py scripts
```

Set the root without editing any code:

```bash
export RIGNET_ROOT=/path/to/RIG-Net
```

No script contains a hard-coded absolute path.
