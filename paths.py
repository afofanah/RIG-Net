"""paths.py - every filesystem location used by the pipeline, defined once.

Project root defaults to the path below; override it without editing code by setting
the RIGNET_ROOT environment variable, e.g.

    export RIGNET_ROOT=/some/other/place
"""
import os
from pathlib import Path

PROJECT = Path(os.environ.get("RIGNET_ROOT", "/Users/s5273738/RIG-Net"))

CODE    = PROJECT / "ebola_gnn"     # the scripts, data/ and results/ live here
DATA    = CODE / "data"             # model-ready arrays and boundary files
RESULTS = CODE / "results"          # run records and derived tables
EXTERNAL= CODE / "external"         # genomes, metadata, source files
UPLOADS = PROJECT / "uploads"       # raw inputs (WHO package, shapefiles)
PAPER   = PROJECT / "paper"         # LaTeX source
FIG     = PAPER / "figs"            # figures written by the paper_*.py scripts

for _d in (DATA, RESULTS, FIG):
    _d.mkdir(parents=True, exist_ok=True)
