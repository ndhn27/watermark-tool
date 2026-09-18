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
  - An optional invisible backup mark (--invisible, images only), a simple
    LSB (least-significant-bit) payload spread across the image. It is NOT
    robust to recompression, resizing, or screenshots (those need
    DCT/frequency-domain or learned watermarking, a much bigger project) -
    it's meant only as a lightweight fallback: if the visible grid is
    cropped or painted over on a losslessly-saved copy, this can still
    confirm provenance. Use extract_invisible.py to read it back.
  - Provenance metadata (images only): --author / --copyright get written
    into real EXIF (JPEG, via piexif) or PNG text chunks, so tools and
    platforms that read standard metadata see attribution even if they
    never look at the pixels at all. This is a separate, complementary
    signal - anyone can strip metadata, so don't rely on it alone.

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

    # read back the invisible mark later
    python3 extract_invisible.py output.png

Setup:
    pip install -r ../requirements.txt
    # or: pip install pillow numpy piexif opencv-python-headless tqdm --break-system-packages
    # video output with audio also needs an ffmpeg binary on PATH
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

from PIL import Image

import video as video_mod
import wmcore

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # noqa: D401 - simple passthrough fallback
        return iterable


# ---------------------------------------------------------------------------
# Still images
# ---------------------------------------------------------------------------

def apply_watermark(
    input_path,
    output_path,
    text,
    opacity=90,
    font_size=36,
    angle=30,
    spacing=250,
    font_path=None,
    seed=None,
    blend="alpha",
    ink=0.32,
    invisible=False,
    invisible_text=None,
    author=None,
    copyright_text=None,
):
    """Watermark a single still image. Also the per-file unit of work for
    the parallel --batch pool below.
    """
    base = Image.open(input_path).convert("RGBA")
    watermark_layer = wmcore.create_watermark_layer(
        base.size, text, font_size, opacity, angle, spacing, font_path, seed=seed
    )
    watermarked = wmcore.composite(base, watermark_layer, mode=blend, ink=ink)

    if invisible:
        if output_path.lower().endswith((".jpg", ".jpeg")):
            print(
                f"Warning: --invisible payload requires a lossless format; "
                f"{output_path} is JPEG and recompression will destroy it. "
                f"Consider a .png output instead."
            )
        watermarked = wmcore.embed_invisible(watermarked, invisible_text or text)

    wmcore.save_with_metadata(watermarked, output_path, author=author, copyright_text=copyright_text)


def _process_one_image(job):
    """Worker for the batch ProcessPoolExecutor - one image per call. Kept
    at module level (not a closure) so it's picklable to worker processes;
    combined with the __main__ guard below, this is the standard safe
    pattern for multiprocessing in a directly-run script (see wmcore.py's
    docstring for the full reasoning).
    """
    in_path, out_path, kwargs = job
    try:
        apply_watermark(in_path, out_path, **kwargs)
        return (in_path, True, None)
    except Exception as exc:  # report failure, don't crash the whole batch
        return (in_path, False, str(exc))


def _stable_file_seed(base_seed, fname):
    """Deterministic per-file seed offset for batches, so every file gets
    its own jitter pattern instead of an identical (and itself detectable)
    one, while still being reproducible run-to-run for a given --seed.
    Thin wrapper around wmcore.derive_seed() - see there for why crc32.
    """
    return wmcore.derive_seed(base_seed, fname)


# ---------------------------------------------------------------------------
# Dispatch (single file: image or video)
# ---------------------------------------------------------------------------

def apply_watermark_any(input_path, output_path, image_kwargs, video_kwargs, workers):
    ext = os.path.splitext(input_path)[1].lower()
    if ext in wmcore.VIDEO_EXTS:
        if image_kwargs.get("invisible") or image_kwargs.get("author") or image_kwargs.get("copyright_text"):
            print(f"Note: --invisible/--author/--copyright are image-only; ignored for {input_path}.")
        video_mod.apply_watermark_video(input_path, output_path, workers=workers, **video_kwargs)
    else:
        apply_watermark(input_path, output_path, **image_kwargs)
        print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply a diagonal grid watermark to an image or video, resistant to cropping and inpainting."
    )
    parser.add_argument("input", help="Input image/video file (or folder if using --batch)")
    parser.add_argument("output", help="Output image/video file (or folder if using --batch)")
    parser.add_argument("--text", required=True, help="Watermark text, e.g. @yourhandle or a group/brand name")
    parser.add_argument("--opacity", type=int, default=90, help="Base opacity, 0-255 (default: 90, ~35%%)")
    parser.add_argument("--font-size", type=int, default=36, help="Font size (default: 36)")
    parser.add_argument("--angle", type=int, default=30, help="Base rotation angle in degrees (default: 30)")
    parser.add_argument("--spacing", type=int, default=250, help="Spacing between watermark tiles, in px (default: 250)")
    parser.add_argument("--font-path", default=None, help="Optional path to a custom .ttf font file")
    parser.add_argument("--batch", action="store_true", help="Process a whole input folder into an output folder")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel workers: files-in-parallel for --batch images, frames-in-parallel for a video (default: CPU count)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for tile jitter (reuse it to reproduce the same pattern)")
    parser.add_argument(
        "--no-shared-memory", dest="use_shared_memory", action="store_false",
        help="Video only: disable the shared-memory frame transport and fall back to the slower "
             "pickle-per-frame path (useful if shared memory isn't available in your environment)",
    )
    parser.add_argument(
        "--jitter-refresh-seconds", type=float, default=4.0,
        help="Video only: regenerate the watermark's jitter pattern (new random tile "
             "positions/opacity/rotation) every N seconds of output instead of using one "
             "static pattern for the whole clip (default: 4.0). A single unchanging pattern "
             "stamped over moving footage is itself averageable out via a per-pixel median "
             "across enough frames, the same collusion/averaging attack the README describes "
             "for multiple images sharing a seed - just within one file instead of across "
             "several. Pass 0 to disable and use one static pattern for the whole video "
             "(old behavior; only meaningful with --seed set, for exact reproducibility).",
    )
    parser.add_argument("--blend", choices=["alpha", "multiply", "overlay"], default="alpha", help="How the mark mixes with the base image (default: alpha)")
    parser.add_argument(
        "--ink", type=float, default=0.32,
        help="Darkness of the mark for multiply/overlay blend, 0-1 (default: 0.32). "
             "For --blend overlay, 0.5 is the mathematical no-op point (the mark "
             "becomes invisible) - stay clearly above or below 0.5. For multiply, "
             "smaller values give a darker mark; 1.0 is the no-op point there instead.",
    )
    parser.add_argument("--invisible", action="store_true", help="Also embed a hidden LSB backup mark (images only, save as PNG - JPEG destroys it)")
    parser.add_argument("--invisible-text", default=None, help="Text for the hidden mark, defaults to --text")
    parser.add_argument("--author", default=None, help="Author name to embed in EXIF/PNG metadata (images only)")
    parser.add_argument("--copyright", dest="copyright_text", default=None, help="Copyright string to embed in EXIF/PNG metadata (images only)")
    args = parser.parse_args()

    image_kwargs = dict(
        text=args.text, opacity=args.opacity, font_size=args.font_size, angle=args.angle,
        spacing=args.spacing, font_path=args.font_path, seed=args.seed, blend=args.blend, ink=args.ink,
        invisible=args.invisible, invisible_text=args.invisible_text, author=args.author,
        copyright_text=args.copyright_text,
    )
    video_kwargs = dict(
        text=args.text, opacity=args.opacity, font_size=args.font_size, angle=args.angle,
        spacing=args.spacing, font_path=args.font_path, seed=args.seed, blend=args.blend, ink=args.ink,
        use_shared_memory=args.use_shared_memory, jitter_refresh_seconds=args.jitter_refresh_seconds,
    )
    workers = args.workers or os.cpu_count() or 1

    if args.batch:
        os.makedirs(args.output, exist_ok=True)
        files = sorted(
            f for f in os.listdir(args.input)
            if f.lower().endswith(wmcore.IMAGE_EXTS + wmcore.VIDEO_EXTS)
        )
        if not files:
            print("No images or videos found in the input folder.")
            return

        image_files = [f for f in files if f.lower().endswith(wmcore.IMAGE_EXTS)]
        video_files = [f for f in files if f.lower().endswith(wmcore.VIDEO_EXTS)]

        if image_files:
            jobs = []
            for fname in image_files:
                kwargs = dict(image_kwargs)
                if args.seed is not None:
                    kwargs["seed"] = _stable_file_seed(args.seed, fname)
                jobs.append((os.path.join(args.input, fname), os.path.join(args.output, fname), kwargs))

            print(f"Watermarking {len(jobs)} image(s) across {workers} worker(s)...")
            failures = []
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for in_path, ok, err in tqdm(pool.map(_process_one_image, jobs), total=len(jobs), unit="img"):
                    if not ok:
                        failures.append((in_path, err))
            for in_path, err in failures:
                print(f"  FAILED: {in_path}: {err}")
            print(f"Done: {len(jobs) - len(failures)}/{len(jobs)} image(s) saved to {args.output}")

        for fname in video_files:
            kwargs = dict(video_kwargs)
            if args.seed is not None:
                kwargs["seed"] = _stable_file_seed(args.seed, fname)
            in_path = os.path.join(args.input, fname)
            out_path = os.path.join(args.output, fname)
            print(f"Watermarking video: {fname} (parallel across {workers} worker(s) for frames)")
            video_mod.apply_watermark_video(in_path, out_path, workers=workers, **kwargs)
    else:
        apply_watermark_any(args.input, args.output, image_kwargs, video_kwargs, workers)


if __name__ == "__main__":
    main()
