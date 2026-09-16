# Mộc — Diagonal Grid Watermark

A small tool that stamps images with a **tiled, diagonal grid watermark** instead of
a single corner logo. Comes in two flavors:

- **`cli/watermark.py`** — a Python script for batch-processing folders of images.
- **`web/index.html`** — a single-file, no-build browser tool with live preview
  (nothing is uploaded anywhere; it all runs on-canvas in your browser).

## Why a grid instead of a corner logo?

A logo tucked in one corner is gone the moment someone crops it out, and small,
localized AI inpainting can erase it in one pass. A watermark tiled diagonally
across the *entire* image is a different problem to remove: getting rid of it
cleanly means repainting almost the whole image, which is far more work and
tends to leave visible artifacts or blur — especially over detailed areas like
faces or text.

## Removal-resistance features (CLI)

A perfectly regular grid still has weaknesses: it has a strong signature in
the frequency domain (FFT) that can be estimated and subtracted, especially
across many images stamped with the same pattern, and a flat white layer can
in principle be separated from the base image. The CLI addresses both:

- **Per-tile jitter** — every tile gets its own random position offset,
  opacity, and rotation, derived from `--seed` (reuse the same seed to
  reproduce an identical pattern; omit it for a fresh one each run). This
  breaks the grid's periodicity so it's much harder to model out.
- **Blend modes** (`--blend multiply|overlay`) — instead of a flat
  alpha-over, the mark mixes into the base image's own pixel values
  (`--ink` controls how strongly), so it can't be cleanly separated as its
  own layer.
- **Invisible backup mark** (`--invisible`) — a simple LSB (least
  significant bit) payload spread across the image, readable back with
  `cli/extract_invisible.py`. This is a lightweight fallback only — it does
  **not** survive recompression, resizing, or screenshots (that needs
  DCT/frequency-domain or learned watermarking, a much larger project) —
  but it can confirm provenance on a losslessly-saved copy even if the
  visible grid gets cropped or painted over. Save as PNG when using it.
- **Provenance metadata** (`--author`, `--copyright`) — written into real
  EXIF (JPEG, via `piexif`) or PNG text chunks, so tools that read standard
  metadata see attribution even without looking at the pixels. Anyone can
  strip metadata, so treat this as a complementary signal, not the main
  defense.

## Web tool

Just open `web/index.html` in a browser (or serve it as a static page, see
below). Drag in an image, tune text / opacity / density / font size / angle /
color, and download a PNG or JPEG at the original resolution.

### Run it locally

```bash
# no build step — just open the file
open web/index.html        # macOS
xdg-open web/index.html    # Linux
```

### Host it (e.g. GitHub Pages)

Since it's a static, single HTML file, you can publish it directly with GitHub
Pages: **Settings → Pages → Deploy from a branch**, and point it at the `web/`
folder (or copy `index.html` to the repo root / a `docs/` folder, depending on
how you want Pages configured).

## CLI tool

Requires [Pillow](https://python-pillow.org/).

```bash
pip install -r requirements.txt
# or: pip install pillow --break-system-packages

# single image
python3 cli/watermark.py input.jpg output.jpg --text "@yourhandle"

# whole folder (batch), reproducible jitter pattern
python3 cli/watermark.py input_folder/ output_folder/ --text "@yourhandle" --batch --seed 42

# texture-aware blend + invisible backup + provenance metadata
python3 cli/watermark.py input.jpg output.png --text "@yourhandle" \
    --blend multiply --invisible --author "Jane Doe" --copyright "(c) 2026 Jane Doe"

# read back the invisible mark later
python3 cli/extract_invisible.py output.png
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--text` | *(required)* | Watermark text, e.g. `@yourhandle` |
| `--opacity` | `90` | Base opacity, 0–255 |
| `--font-size` | `36` | Font size in px |
| `--angle` | `30` | Base rotation angle in degrees |
| `--spacing` | `250` | Spacing between watermark tiles, in px |
| `--font-path` | *(auto)* | Path to a custom `.ttf` font |
| `--batch` | off | Treat input/output as folders |
| `--seed` | random | Seed for per-tile jitter (reuse to reproduce the same pattern) |
| `--blend` | `alpha` | `alpha`, `multiply`, or `overlay` — how the mark mixes with the base image |
| `--ink` | `0.32` | Mark darkness for `multiply`/`overlay`, 0–1 |
| `--invisible` | off | Also embed a hidden LSB backup mark (save as PNG) |
| `--invisible-text` | *(same as `--text`)* | Text for the hidden mark |
| `--author` | — | Author name, embedded in EXIF/PNG metadata |
| `--copyright` | — | Copyright string, embedded in EXIF/PNG metadata |

## Project structure

```
.
├── cli/
│   ├── watermark.py           # Python/Pillow CLI: jitter, blend modes, invisible mark, metadata
│   └── extract_invisible.py   # Reads back the hidden LSB payload
├── web/
│   └── index.html     # Browser tool, canvas-based, runs entirely client-side
├── requirements.txt
└── README.md
```

## License

MIT — see [LICENSE](LICENSE).
