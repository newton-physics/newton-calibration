# Publishing calibration results

These are retained instructions for the original TaskTuning results site.
They do not configure or authorize publication from newton-calibration; the links below
refer to the original repository. The seed retains the static files as historical evidence.

The repository contains a static GitHub Pages site. It needs no Python, Newton, Node, or simulation at serving time.

## One-time repository setting

After the files are pushed, a repository administrator or maintainer enables publication:

1. Open [repository Pages settings](https://github.com/SamdianJ/TaskTuning/settings/pages).
2. Under **Build and deployment**, choose **Deploy from a branch**.
3. Select **main** and **/(root)**, then save.
4. Wait for the **pages build and deployment** run in [Actions](https://github.com/SamdianJ/TaskTuning/actions) to finish.
5. Open [the results homepage](https://samdianj.github.io/TaskTuning/).

The files alone do not enable Pages. SSH push access does not provide permission to change Pages settings through the REST API. Before enablement and the first successful deployment, the site links may return 404.

GitHub documents this setup in [Configuring a publishing source](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site). The root `.nojekyll` file selects static serving without Jekyll processing; `index.html` provides the homepage. Subsequent pushes to main trigger publication through GitHub's branch deployment mechanism; no custom workflow is required.

## Public entry points

- [Results homepage](https://samdianj.github.io/TaskTuning/)
- [English report](https://samdianj.github.io/TaskTuning/examples/cable_calibration/reports/cable_calibration_en.html)
- [Chinese report](https://samdianj.github.io/TaskTuning/examples/cable_calibration/reports/cable_calibration_zh.html)

The reports retain their existing paths and language switch. Source references link to GitHub so Markdown and JSON evidence remain accessible from the website. Figures/styles are embedded for offline reading; source references need internet access.

## Update the preview

The PNG preview is derived from archived values, not a fresh simulation or a hand-drawn result. Regenerate it from the repository root:

```bash
python -m pip install matplotlib
python -m examples.cable_calibration.verify
python -m examples.cable_calibration.render_preview
```

Matplotlib is only a preview-generation dependency, not a core or deployed-site dependency. The renderer reads processed observations, archived traces, and expected metrics. It plots the no-plug coordinate RMSE for A/B/B*/C and the measured/predicted z coordinate at 150 mm along the cable, using the scoring mask.

Keep uncertainty and acceptance statements when updating figures: A → B is not an angle-only ablation; B* and C share the dynamic objective; plugged validation remains unresolved. Update report export hashes in archive/provenance.json whenever an HTML report changes.

## Local preview and checks

From the repository root:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Visit `http://127.0.0.1:8000/`. Check the desktop and narrow layouts, both language links, chart toggles in the report, and source links to GitHub. Stop the server when finished.

After pushing, confirm the Pages deployment succeeds and check the homepage, both reports, and preview image over HTTPS. A successful Git push alone does not prove the website is live.
