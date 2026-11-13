<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: CC-BY-4.0
-->

# Viewing and maintaining calibration reports

Reports and their supporting evidence are bundled in the repository. This repository
does not configure a hosted report site or an automatic publishing workflow.

## Local preview

From the repository root:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Open `http://127.0.0.1:8000/` for the results index, then choose the English or Chinese
report. Check narrow and desktop layouts, language links, chart toggles, and source links.
Stop the server when finished.

- [Results index](../index.html)
- [English report](../examples/cable_calibration/reports/cable_calibration_en.html)
- [Chinese report](../examples/cable_calibration/reports/cable_calibration_zh.html)
- [Case and reproduction guide](../examples/cable_calibration/README.md)

The HTML reports embed their figures, plots, and styles and can also be opened directly
after download. Keep both files in the same folder for language switching. Source links
are relative to the repository, so browsing the evidence requires the surrounding checkout.

## Update the preview image

The PNG preview is generated from archived values. Regenerate it from the repository root:

```bash
python -m pip install matplotlib
python -m examples.cable_calibration.verify
python -m examples.cable_calibration.render_preview
```

Matplotlib is a preview-generation dependency, not a core runtime dependency. The renderer
reads processed observations, archived traces, and expected metrics. It plots the no-plug
coordinate RMSE for B* and C and the measured and predicted z position at 150 mm using the scoring mask.
Both models were fitted to the same no-plug release recording. Use the common objective for the B* to C comparison.
Git tracks report revisions. Keep historical run and original evidence fingerprints in
`examples/cable_calibration/archive/provenance.json` unchanged when editing reports.
Check links and run the archive verifier before submitting the change.
