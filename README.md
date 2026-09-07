NBT scheduling project
======================

This repository keeps notebook exploration separate from reusable pipeline code.

Recommended workflow:

1. Explore ideas in `notebooks/`.
2. Move repeatable logic into `src/nbt_pipeline/`.
3. Import those functions back into notebooks for analysis and plots.

Current pipeline structure:

```text
src/nbt_pipeline/
├── config.py
├── main.py
├── preprocessing/
│   ├── load.py        # Load the raw Excel dataset
│   ├── clean.py       # Missingness and text-cleaning helpers
│   ├── codes.py       # Readable labels for coded columns
│   ├── specialty.py   # Specialty extraction from SessionIDdesc
│   ├── features.py    # Duration, overrun, note flags, and engineered features
│   └── pipeline.py    # Full preprocessing pipeline
└── outputs/
    └── export.py      # Save CSV/XLSX outputs
```

Run the deterministic preprocessing pipeline:

```powershell
$env:PYTHONPATH='src'
python -m nbt_pipeline.main
```

The output is saved locally to:

```text
data/processed/nbt_smallset_preprocessed.xlsx
```

## Modelling

Predicting how long an operation actually takes, using only what is known
before the patient goes into theatre, and benchmarking that against the
time the hospital books.

### Running it

```bash
export PYTHONPATH=src
python -m nbt_pipeline.modelling_v3.run_all              # everything
python -m nbt_pipeline.modelling_v3.run_all --skip-slow  # without the long searches
python -m nbt_pipeline.modelling_v3.run_all --quick      # fast stages only
```

Charts and result tables are written to `data/modelling_v2/plots` and
`data/modelling_v3/plots`.

### Layout

`src/nbt_pipeline/modelling` holds the first round: the data splits, the
five models, and the first set of charts.

`src/nbt_pipeline/modelling_v2` holds the second round, which added
historical averages per procedure and surgeon, pulled the stated duration
out of the theatre notes, capped very long cases, and retuned everything.

`src/nbt_pipeline/modelling_v3` holds the checks on how the results were
measured: grouped folds against random folds, why some folds score higher
than others, and the analysis of planning error.

### Models

Ridge regression, random forest, XGBoost, support vector regression, and a
model that reads the free text theatre notes. Every one is tuned with grid
search under cross validation, and scored once on a held out test set.

### What we found

Most operations finish early rather than late. 58% come in under the booked
time, and across the dataset that adds up to around 314,000 theatre minutes
booked but not used, which is roughly 654 full theatre days.

Bookings are template values rather than estimates. Six round numbers
account for 79% of all bookings, with 60 minutes alone used a third of the
time.

XGBoost predicts realised duration best, at 25 minutes average error
against the hospital's 41.

Random folds flatter the result. Holding out whole surgeons or whole
procedures drops the score by 0.03 to 0.05, because random folds let the
same surgeon appear on both sides of the split.

How much a model appears to know depends entirely on what it is allowed to
see. Using only what is known at booking, the ceiling is around 0.73.
Adding theatre timings takes it past 0.99, but that information does not
exist when the list is being built.

### Tests

```bash
python -m pytest tests/ -q
```
