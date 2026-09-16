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

# whole folder (batch)
python3 cli/watermark.py input_folder/ output_folder/ --text "@yourhandle" --batch
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--text` | *(required)* | Watermark text, e.g. `@yourhandle` |
| `--opacity` | `90` | Opacity, 0–255 |
| `--font-size` | `36` | Font size in px |
| `--angle` | `30` | Rotation angle in degrees |
| `--spacing` | `250` | Spacing between watermark instances, in px |
| `--font-path` | *(auto)* | Path to a custom `.ttf` font |
| `--batch` | off | Treat input/output as folders |

## Project structure

```
.
├── cli/
│   └── watermark.py   # Python/Pillow CLI, supports single-file and batch mode
├── web/
│   └── index.html     # Browser tool, canvas-based, runs entirely client-side
├── requirements.txt
└── README.md
```

## License

MIT — see [LICENSE](LICENSE).
