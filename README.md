# Mộc — Diagonal Grid Watermark

A small tool that stamps images **and video** with a **tiled, diagonal grid
watermark** instead of a single corner logo. Comes in two flavors:

- **`cli/watermark.py`** — a Python script for images and video, single
  files or whole folders, parallelized across CPU cores with a progress bar.
- **`web/index.html`** — a single-file, no-build browser tool with live
  preview and multi-image batch export (nothing is uploaded anywhere; it
  all runs on-canvas in your browser).

## Why a grid instead of a corner logo?

A logo tucked in one corner is gone the moment someone crops it out, and
small, localized AI inpainting can erase it in one pass. A watermark tiled
diagonally across the *entire* frame is a different problem to remove:
getting rid of it cleanly means repainting almost the whole image (or every
frame of a video), which is far more work and tends to leave visible
artifacts or blur — especially over detailed areas like faces or text.

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
  own layer. Applies to both images and video frames.
- **Invisible backup mark** (`--invisible`, images only) — a simple LSB
  (least significant bit) payload spread across the image, readable back
  with `cli/extract_invisible.py`. This is a lightweight fallback only — it
  does **not** survive recompression, resizing, or screenshots (that needs
  DCT/frequency-domain or learned watermarking, a much larger project) —
  but it can confirm provenance on a losslessly-saved copy even if the
  visible grid gets cropped or painted over. Save as PNG when using it.
- **Provenance metadata** (`--author`, `--copyright`, images only) —
  written into real EXIF (JPEG, via `piexif`) or PNG text chunks, so tools
  that read standard metadata see attribution even without looking at the
  pixels. Anyone can strip metadata, so treat this as a complementary
  signal, not the main defense.

## Video support

Point the CLI at a video file (`.mp4`, `.mov`, `.m4v`, `.avi`, `.mkv`,
`.webm`) and it's detected automatically — same command, no separate flag.
The watermark grid is built once and stamped onto every frame; frame
compositing is parallelized across `--workers` processes so it makes real
use of multiple cores.

If `ffmpeg` is on your `PATH`, output is re-encoded to H.264 with the
**original audio track muxed back in**, in a single pass. Without `ffmpeg`,
it falls back to OpenCV's own writer — the video still gets watermarked,
but the output has **no audio** and uses a much less efficient codec. Get
`ffmpeg` from [ffmpeg.org](https://ffmpeg.org) (or `apt install ffmpeg` /
`brew install ffmpeg`) for real use.

`--invisible`, `--author`, and `--copyright` are image-only and are ignored
(with a note) for video input.

## Batch mode & performance

`--batch` processes a whole folder — images and videos can be mixed in the
same folder. Images are processed **in parallel** across `--workers`
processes (default: your CPU count) with a live progress bar; each video is
still processed one at a time, but its own frames are parallelized
internally the same way. A failure on one file (corrupt image, unreadable
video) is reported and skipped rather than stopping the whole batch.

Every file in a batch gets its own jitter pattern (deterministically derived
from `--seed` + filename), so a fixed `--seed` still gives you reproducible
output without every file in the batch sharing an identical watermark
pattern.

## Web tool

Open `web/index.html` in a browser (or serve it as a static page, see
below). Drag in **one or more** images, tune text / opacity / density / font
size / angle / color, and:

- **Download this image** — the currently-selected image only.
- **Download all as .zip** — watermarks every loaded image with the same
  settings and bundles them into a single `.zip`, with a progress bar while
  it works. The zip is built entirely client-side (no library, no upload).

Switch between loaded images using the thumbnail strip above the preview;
each keeps its own original resolution on export. Video isn't supported in
the browser tool — use the CLI for that.

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

```bash
pip install -r requirements.txt
# or: pip install pillow numpy opencv-python-headless tqdm --break-system-packages
# (video output with audio also needs an ffmpeg binary on PATH)
```

`piexif` (in `requirements.txt`) is optional: it's only needed for
`--author`/`--copyright` metadata on **JPEG** output. If it's missing,
those flags are silently skipped with a printed note — everything else,
including PNG metadata, works without it.

If **no TrueType font is found** (see `--font-path` below): with
Pillow >= 10.1 the tool falls back to Pillow's built-in scalable font, so
`--font-size` still works, just with a different typeface than DejaVu/Arial.
On older Pillow, the fallback is a tiny fixed-size bitmap font that ignores
`--font-size` (a warning is printed). Either way, pass `--font-path
/path/to/some.ttf` for full control (or `brew install --cask
font-dejavu-sans` / your distro's `fonts-dejavu` package to match the
Linux default look).

```bash
# single image
python3 cli/watermark.py input.jpg output.jpg --text "@yourhandle"

# single video (audio kept automatically if ffmpeg is installed)
python3 cli/watermark.py input.mp4 output.mp4 --text "@yourhandle"

# whole folder (images and videos mixed), parallel, reproducible jitter
python3 cli/watermark.py input_folder/ output_folder/ --text "@yourhandle" \
    --batch --seed 42 --workers 8

# texture-aware blend + invisible backup + provenance metadata (images)
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
| `--font-path` | *(auto)* | Path to a custom `.ttf` font. Auto-detect tries DejaVu Sans Bold (Linux), Arial Bold (macOS), then Arial Bold (Windows), in that order; see `DEFAULT_FONT_PATHS` in `cli/wmcore.py` |
| `--batch` | off | Treat input/output as folders (images + video, mixed) |
| `--workers` | CPU count | Parallel workers: files-in-parallel for `--batch` images, frames-in-parallel for a video |
| `--seed` | random | Seed for per-tile jitter (reuse to reproduce the same pattern) |
| `--blend` | `alpha` | `alpha`, `multiply`, or `overlay` — how the mark mixes with the base pixels |
| `--ink` | `0.32` | Mark darkness for `multiply`/`overlay`, 0–1 |
| `--invisible` | off | Also embed a hidden LSB backup mark (images only, save as PNG) |
| `--invisible-text` | *(same as `--text`)* | Text for the hidden mark |
| `--author` | — | Author name, embedded in EXIF/PNG metadata (images only) |
| `--copyright` | — | Copyright string, embedded in EXIF/PNG metadata (images only) |
| `--no-shared-memory` | off | Video only: disable the shared-memory frame transport, fall back to a slower pickle-per-frame path (useful if shared memory isn't available in your environment, e.g. some sandboxes without `/dev/shm`) |

## Project structure

```
.
├── cli/
│   ├── wmcore.py               # Shared grid/blend/LSB/metadata logic (images + video)
│   ├── watermark.py            # CLI entry: images, batch + multiprocessing, dispatches video
│   ├── video.py                # Video pipeline: parallel frame workers, ffmpeg mux/encode
│   └── extract_invisible.py    # Reads back the hidden LSB payload
├── tests/                      # pytest unit tests for the blend math + LSB round trip
├── web/
│   └── index.html      # Browser tool: multi-image batch, zip export, runs entirely client-side
├── requirements.txt
├── requirements-dev.txt        # requirements.txt + pytest
└── README.md
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/
```

Covers the blend-mode math (`wmcore.blend_arrays`, all three modes) and the
invisible-LSB embed/extract round trip - the two places most likely to
silently regress since neither has an obvious "it crashed" failure mode
(a wrong blend or a broken payload just *looks* fine at a glance).

## License

MIT — see [LICENSE](LICENSE).
