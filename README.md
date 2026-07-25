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
