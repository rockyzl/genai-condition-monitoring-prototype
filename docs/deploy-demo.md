# Deploy the public web demo

How to publish this repo as a one-click public web demo and embed it in
`sciencesloop.com/agent`. Primary target: **Hugging Face Spaces (Streamlit
SDK)**; secondary: **Streamlit Community Cloud**.

## What runs on boot

The repo intentionally commits **no data, no model, no processed artifacts** —
`data/`, `models/*.joblib`, and the pipeline journal are all gitignored. So a
fresh deploy has nothing to serve yet. The root entry point handles this:

- **`streamlit_app.py`** (repo root, auto-detected by both platforms) checks
  whether the required artifacts exist
  (`data/processed/test_predictions.csv`, `models/rul_baseline.joblib`, the
  evidence dir, `reports/metrics_model.json`).
  - **Missing** (cold start) → it renders a bilingual bootstrap screen and runs
    `scripts/bootstrap_demo.py`: download NASA C-MAPSS (~12 MB) → extract FD001
    → run the 10-stage pipeline in-process. ~**1 minute** on a free CPU box
    (measured: 49 s end-to-end, download + RandomForest train + predict +
    evidence + diagnose + eval). Then it reruns.
  - **Present** (warm) → it delegates straight to the real app
    (`src/app/streamlit_app.py`), executed as `__main__` via `runpy` so that app
    keeps ownership of `st.set_page_config`.

You can build the artifacts ahead of time too:

```bash
.venv/bin/python scripts/bootstrap_demo.py          # download + full pipeline
.venv/bin/python scripts/bootstrap_demo.py --force  # re-download + rerun all stages
```

It is idempotent — a second run is a no-op (provenance skips), so it is safe to
call on every boot.

---

## A. Hugging Face Spaces (primary)

HF Spaces still supports a native Streamlit SDK (`sdk: streamlit`). It serves on
port **8501 only** (do not override the port in a `config.toml`).

1. **Create the Space.** <https://huggingface.co/new-space> → pick **Streamlit**
   as the SDK, CPU basic (free) hardware, public. HF pre-fills `sdk_version`
   with its latest supported Streamlit — **keep that value.**
2. **Set the Space README front-matter.** The Space's root `README.md` must
   begin with a YAML block. Use `README_SPACE.md` from this repo — either rename
   it to `README.md` on the Space, or prepend its front-matter to your existing
   `README.md`. The important keys:

   ```yaml
   ---
   title: Condition Monitoring Agent
   emoji: 🛠️
   colorFrom: blue
   colorTo: indigo
   sdk: streamlit
   sdk_version: "1.40.0"   # keep HF's pre-filled latest-supported value
   app_file: streamlit_app.py
   pinned: false
   suggested_hardware: cpu-basic
   ---
   ```

   > `sdk_version` note: not every Streamlit version is HF-supported. The app was
   > tested on Streamlit 1.59.0 and targets `>= 1.30`; use whatever recent version
   > HF offers — the app is not version-sensitive. Leave `disable_embedding`
   > unset (embedding is allowed by default).
3. **Push this repo to the Space** (Spaces are git repos). `requirements.txt` is
   sufficient as-is — no extra packages are needed. The dataset download uses the
   Python stdlib (`urllib`), and `threadpoolctl` is pulled in transitively by
   scikit-learn. HF installs Streamlit via `sdk_version`; keeping
   `streamlit>=1.30` in `requirements.txt` is compatible (do not pin a
   conflicting exact version).
4. **First visit bootstraps (~1 min).** The Space build + server start are fast;
   the download+train runs inside the *first* user session (they watch the boot
   screen for ~1 min). Every later visit to the same warm container is instant.

### Honest caveat — cold starts always re-bootstrap

HF Spaces storage is **ephemeral**, and the persistent-storage feature has been
**retired** (HF: *"the persistent storage feature is no longer available"*).
That means:

- On free CPU tier the Space **sleeps after ~48 h of inactivity**; the next visit
  cold-starts and **re-runs the ~1-minute bootstrap**.
- A rebuild/restart also re-bootstraps.
- **Mitigation:** none needed beyond accepting the ~1 min — we commit no data, so
  there is nothing to persist, and the bootstrap is cheap and deterministic.
  For a truly instant embed, pair the iframe with the static fallback (below) so
  visitors see a screenshot + "Launch live demo" while the Space wakes.

---

## B. Streamlit Community Cloud (secondary)

1. Go to <https://share.streamlit.io> → **New app** → connect the GitHub repo.
2. Set **Main file path** to `streamlit_app.py`, branch to your default branch.
3. Deploy. Same bootstrap-on-first-boot behavior. Community Cloud also sleeps on
   inactivity and rebuilds ephemerally, so the ~1-minute bootstrap recurs on cold
   starts.

> Resource note: Community Cloud gives ~1 GB RAM. Peak usage during the RF train
> is roughly ~1 GB (data + 200-tree forest), so it fits but is tighter than HF.
> If a boot ever OOMs, lower `rf_params.n_estimators` in `config/pipeline.yaml`
> (e.g. 200 → 120) — accuracy barely moves and memory drops.

---

## C. Embed in sciencesloop.com/agent

Every Space is reachable at `https://<username>-<space-name>.hf.space`
(the profile/space names are lowercased and dashed). Add `?embed=true` to strip
the Streamlit chrome (menu + footer) for a slim embed.

### Fixed-height iframe (simplest, recommended)

```html
<iframe
  src="https://<username>-condition-monitoring-agent.hf.space/?embed=true"
  title="Condition Monitoring Agent — live demo"
  width="100%"
  height="900"
  style="border:0; border-radius:12px; max-width:1180px; margin:0 auto; display:block;"
  loading="lazy"
  allow="clipboard-write; fullscreen"
  referrerpolicy="no-referrer-when-downgrade">
</iframe>
```

- Streamlit iframes do **not** auto-size to content, so give a generous fixed
  `height` (800–1000 px) and let the app scroll inside it.
- `max-width` + `margin:auto` keeps it centered and readable on wide screens;
  `width:100%` makes it responsive down to mobile.
- `loading="lazy"` avoids paying the Space wake-up cost until the iframe scrolls
  into view.

### Optional: auto-resizing iframe

Streamlit supports iframe auto-resize (since 1.17) via `iframe-resizer`. Only use
this if you control the parent page and can load the external script:

```html
<iframe id="cm-demo" src="https://<username>-condition-monitoring-agent.hf.space"
        frameborder="0" width="100%" height="900"></iframe>
<script src="https://cdn.jsdelivr.net/npm/iframe-resizer@4.3.4/js/iframeResizer.min.js"></script>
<script>iFrameResize({ checkOrigin: false }, "#cm-demo")</script>
```

### Static fallback for when the Space sleeps (recommended)

A slept Space shows HF's "building" screen for ~30–60 s on wake, which looks
broken inside an embed. Guard against it: show a **screenshot with a "Launch live
demo ↗" overlay** by default, and only swap in the live iframe on click.

```html
<a class="cm-demo-launch" href="https://<username>-condition-monitoring-agent.hf.space/?embed=true"
   target="_blank" rel="noopener">
  <img src="/img/cm-demo-screenshot.png" alt="Condition Monitoring Agent demo"
       style="max-width:1180px;width:100%;border-radius:12px;" />
  <span>▶ Launch live demo (opens the Space; first load ~1 min while it wakes)</span>
</a>
```

Capture the screenshot once from a warm Space (Decision Inbox view reads best).

---

## D. Resource notes (fits free tiers)

| Item | Value |
|---|---|
| First-boot time (download + full pipeline) | ~1 min on free CPU (measured 49 s) |
| Download size (NASA C-MAPSS zip) | 12.4 MB |
| Trained model (`rul_baseline.joblib`) | ~81 MB, rebuilt on the box (not committed) |
| Peak RAM during RF train | ~1 GB |
| Steady-state RAM (serving) | a few hundred MB |
| HF free tier | 2 vCPU / 16 GB RAM → comfortable |
| Streamlit Community Cloud | ~1 GB RAM → fits, tighter (see B) |

Because the model is trained **on the box** at boot (never committed), there is
no cross-version unpickling risk — whatever scikit-learn the platform installs is
the one that builds and loads the model. The trade-off is the ~1-minute cold
start, which the static fallback hides from embedded visitors.

## Local verification of the deploy entry

```bash
# fresh-clone simulation already validated: download + pipeline + tests green.
.venv/bin/python scripts/bootstrap_demo.py            # build artifacts
.venv/bin/streamlit run streamlit_app.py              # warm → delegates to the app

# exercise the cold-boot screen without a real download:
DEMO_FORCE_BOOTSTRAP=1 DEMO_BOOTSTRAP_DRYRUN=1 \
  .venv/bin/streamlit run streamlit_app.py
```

`DEMO_FORCE_BOOTSTRAP=1` makes the entry treat artifacts as missing;
`DEMO_BOOTSTRAP_DRYRUN=1` renders the boot screen but skips the real
download/pipeline. Both are off in normal use.
