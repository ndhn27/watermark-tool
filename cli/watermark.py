#!/usr/bin/env python3
"""
watermark.py - Apply a tiled, diagonal grid watermark to images and videos,
with several layers of removal-resistance stacked on top of the basic grid.

Why a diagonal grid instead of a single corner logo?
  - A corner logo is gone with a single crop, and small-area AI inpainting
    can erase it easily.
  - A diagonal grid covers the whole image (or every video frame), so
    removing it cleanly means an AI (or a human) essentially has to repaint
    the entire frame. That's far more effort, and it tends to leave
    artifacts or blur, especially over detailed areas (faces, text).

Removal-resistance features:
  - Per-tile jitter (position / opacity / rotation), seeded with --seed.
    A perfectly regular grid has a strong, narrow signature in the
    frequency domain (FFT), which makes it possible to estimate and
    subtract automatically, especially if an attacker has several images
    stamped with the same pattern to average over. Randomizing each tile
    independently breaks that periodicity.
  - Blend modes (--blend multiply|overlay) instead of a flat alpha
    composite. A flat-alpha white watermark sits on its own "layer" that
    can, in principle, be estimated and subtracted. Multiply/overlay mix
    the mark into the local pixel values instead, so removal requires
    reconstructing the underlying image data, not just subtracting a
    constant layer.
  - An optional invisible backup mark (--invisible, images only), in one of
    two schemes (--invisible-mode):
      - lsb (default): a simple least-significant-bit payload spread
        across the image. Fast, no key needed, but NOT robust to
        recompression, resizing, or screenshots - it's meant only as a
        lightweight fallback: if the visible grid is cropped or painted
        over on a losslessly-saved (PNG) copy, this can still confirm
        provenance.
      - spread: a block-DCT spread-spectrum payload that DOES survive
        ordinary JPEG recompression and a block-aligned crop, at the cost
        of needing --ss-key to read it back later (one is generated and
        printed for you if omitted), being slower, and needing a
        reasonably large image. See wmcore.py's module comment above
        embed_spread_spectrum() and the README for the full tradeoffs -
        in particular, --ss-key is a reproducibility key like --seed, not
        an access-control secret.
    Use extract_invisible.py to read either one back.
  - Provenance metadata (images only): --author / --copyright get written
    into real EXIF (JPEG, via piexif) or PNG text chunks, so tools and
    platforms that read standard metadata see attribution even if they
    never look at the pixels at all. This is a separate, complementary
    signal - anyone can strip metadata, so don't rely on it alone.
  - Content-adaptive placement (--adaptive): a lightweight face detector
    (opencv's bundled Haar cascade) plus a local detail-energy signal find
    the faces / high-detail "main subject" regions where a clean
    inpaint-based removal is most likely to leave a visible seam, and the
    grid gets denser (and, for --blend multiply/overlay, darker) exactly
    there via wmcore.compute_sensitivity_map / build_ink_map. See the
    README for the full flag list (--adaptive-strength, --face-cascade,
    --adaptive-preview, etc).

Video support:
  - Any input ending in one of wmcore.VIDEO_EXTS (.mp4, .mov, .m4v, .avi,
    .mkv, .webm) is routed to video.py instead of the image path. The same
    grid/jitter/blend math stamps every frame; frame compositing is
    parallelized across --workers processes, and (if ffmpeg is on PATH)
    the result is re-encoded to H.264 with the original audio track muxed
    back in, in one pass. --invisible/--author/--copyright don't apply to
    video and are ignored with a note.

Batch mode (--batch) parallelizes across files with --workers processes
(images) and shows a progress bar; videos in a batch are still processed
one at a time (each video already parallelizes internally across frames).

Usage:
    # single image
    python3 watermark.py input.jpg output.jpg --text "@yourhandle"

    # single video (audio kept automatically if ffmpeg is installed)
    python3 watermark.py input.mp4 output.mp4 --text "@yourhandle"

    # whole folder (images and videos mixed), parallel, reproducible jitter
    python3 watermark.py in/ out/ --text "@yourhandle" --batch --seed 42 --workers 8

    # texture-aware blend + invisible backup + provenance metadata (images)
    python3 watermark.py input.jpg output.png --text "@yourhandle" \\
        --blend multiply --invisible --author "Jane Doe" --copyright "(c) 2026 Jane Doe"

    # invisible mark that survives JPEG recompression (spread-spectrum)
    python3 watermark.py input.jpg output.jpg --text "@yourhandle" \\
        --invisible --invisible-mode spread --ss-key 20260918

    # content-adaptive placement: denser + darker over faces/main subject
    python3 watermark.py input.jpg output.png --text "@yourhandle" \\
        --blend multiply --adaptive --adaptive-strength 0.8 --adaptive-preview mask.png

    # read back the invisible mark later
    python3 extract_invisible.py output.png
    python3 extract_invisible.py output.jpg --mode spread --ss-key 20260918

Architecture:
    This file is only the CLI shell: it turns argv into a wmapp.WatermarkSpec
    (the flags are generated from wmapp.options.OPTIONS, the same list the
    Python API and the HTTP server use), calls wmapp.prepare() / watermark_file()
    / watermark_batch(), and prints the notes they produce. The pipeline itself
    lives in wmapp/pipeline.py - see there, and wmapp/__init__.py for the layering.

Setup:
    pip install -r ../requirements.txt
    # or: pip install pillow numpy piexif opencv-python-headless tqdm --break-system-packages
    # video output with audio also needs an ffmpeg binary on PATH
"""

import argparse
import os

import wmapp
import wmcore  # noqa: F401  (kept importable from here for callers that used watermark.wmcore)

try:
    from tqdm import tqdm
except ImportError:  # requirements.txt lists tqdm; a missing one just means no progress bar
    tqdm = None


def _print_note(note):
    """How the CLI shows a wmapp.Note: exactly its message, on stdout - which is
    what the pre-wmapp print() calls did."""
    print(note.message)


def apply_watermark(input_path, output_path, text, **options):
    """Backward-compatible wrapper: watermark one still image, keyword options
    as before (opacity, blend, invisible, ss_key, c2pa, ...). New code should
    call wmapp.watermark(), which also resolves a missing ss_key / dev cert for
    you and returns a FileResult; this keeps the old contract - no run-level
    resolution (invisible_mode="spread" without ss_key raises ValueError), and
    warnings printed to stdout."""
    spec = wmapp.WatermarkSpec(text=text, **options)
    wmapp.watermark_file(input_path, output_path, wmapp.Prepared(spec), on_note=_print_note)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Apply a diagonal grid watermark to an image or video, resistant to cropping and inpainting."
    )
    parser.add_argument("input", help="Input image/video file (or folder if using --batch)")
    parser.add_argument("output", help="Output image/video file (or folder if using --batch)")
    wmapp.add_arguments(parser, wmapp.OPTIONS)
    return parser


def _batch_progress():
    """on_progress callback drawing a tqdm bar (or nothing, without tqdm)."""
    if tqdm is None:
        return None
    state = {}

    def progress(done, total):
        bar = state.get("bar")
        if bar is None:
            bar = state["bar"] = tqdm(total=total, unit="img")
        bar.update(done - bar.n)
        if done >= total:
            bar.close()

    return progress


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        spec = wmapp.spec_from_namespace(wmapp.WatermarkSpec, args)
        prepared = wmapp.prepare(spec, batch=args.batch, on_note=_print_note)
    except (wmapp.SpecError, wmapp.UnavailableError) as exc:
        parser.error(str(exc))

    workers = args.workers or os.cpu_count() or 1

    if args.batch:
        wmapp.watermark_batch(
            args.input, args.output, prepared, workers=workers, show_progress=True,
            on_note=_print_note, on_progress=_batch_progress(),
        )
    else:
        wmapp.watermark_file(args.input, args.output, prepared, workers=workers,
                             show_progress=True, on_note=_print_note)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
