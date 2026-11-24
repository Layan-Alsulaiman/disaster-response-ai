# Sample disaster cases

No imagery is committed to this repository. Add 2–3 real xBD cases here.

1. Register and download xBD from https://xview2.org/dataset (the "tier1"/"test" splits are enough; do not process the full dataset).
2. Pick a few tiles with visible damage. Each xBD tile has matching files:
   - `images/<disaster>_<id>_pre_disaster.png`
   - `images/<disaster>_<id>_post_disaster.png`
   - `labels/<disaster>_<id>_post_disaster.json` (optional, for later evaluation only)
3. Copy them into one folder per case, for example:

```
data/samples/
└── hurricane-michael_00000123/
    ├── hurricane-michael_00000123_pre_disaster.png
    ├── hurricane-michael_00000123_post_disaster.png
    └── hurricane-michael_00000123_post_disaster.json   (optional)
```

Only the **post-disaster** image is fed to the model. The pre-disaster image is shown for visual comparison only. The JSON annotations are ground truth and must never be displayed as AI predictions.

xBD is licensed CC BY-NC-SA 4.0.
