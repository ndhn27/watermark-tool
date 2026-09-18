#!/usr/bin/env python3
"""
wmcore.py - Shared watermarking primitives used by watermark.py (images),
video.py (video frames), and their multiprocessing workers.

Keeping this logic in one importable, non-"__main__" module matters for two
reasons:
  1. It's the single source of truth for the actual watermark math, so the
     image path and the video path can never silently drift apart.
  2. Functions handed to multiprocessing.Pool / ProcessPoolExecutor must be
     picklable by reference. On the "spawn" start method (the default on
     Windows and macOS), that means they must live in a plain top-level
     module that child processes can just `import` - not in the __main__
     script, and not as a closure or lambda. Everything reusable across
     processes lives here for exactly that reason.
"""

import math
import os
import random
import zlib

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

try:
    import piexif
except ImportError:
    piexif = None

DEFAULT_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux/WSL2
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",      # macOS (Catalina+, stock)
    "/Library/Fonts/Arial Bold.ttf",                          # macOS (older / MS Office install)
    "C:\\Windows\\Fonts\\arialbd.ttf",                         # Windows fallback
]

VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

_font_fallback_warned = False


def load_font(font_size, font_path=None):
    """Load a TrueType font, trying `font_path` (if given) then the
    platform-default candidates in DEFAULT_FONT_PATHS, in order.

    If none of those exist, we still try PIL's built-in default font at
    `font_size` (Pillow >= 10.1 supports this: `ImageFont.load_default(size=...)`
    returns a real scalable font instead of the old fixed tiny bitmap). Only
    on older Pillow, where that isn't available, do we fall back to the
    classic `load_default()` - which ignores `font_size` and is tiny/blocky
    - and warn once per process about it (this can be called once per
    output file, so a large --batch run shouldn't spam the same warning).
    """
    global _font_fallback_warned
    candidates = [font_path] if font_path else DEFAULT_FONT_PATHS
    for path in candidates:
        if path and os.path.exists(path):
            return ImageFont.truetype(path, font_size)
    try:
        return ImageFont.load_default(size=font_size)
    except TypeError:
        pass  # Pillow < 10.1: load_default() doesn't take a size argument
    if not _font_fallback_warned:
        checked = ", ".join(p for p in candidates if p)
        print(
            f"Warning: no TrueType font found (checked: {checked}) and this Pillow "
            "version's built-in fallback font is fixed-size. Text will be tiny and "
            "won't scale with --font-size. Pass --font-path /path/to/font.ttf, or "
            "upgrade Pillow to >=10.1, to fix this."
        )
        _font_fallback_warned = True
    return ImageFont.load_default()


def derive_seed(base_seed, key):
    """Deterministically derive a distinct-but-reproducible seed from a base
    seed and a key (a filename, a video segment index, etc.), so several
    items sharing one base seed each get their own jitter pattern instead of
    an identical one - which would itself be a single static pattern an
    attacker could average/median out across those items. Returns None
    unchanged when base_seed is None, preserving "fresh randomness every
    run" for callers who never pinned a seed to begin with.

    Uses crc32 rather than the built-in hash(): Python salts string hashing
    per-process by default, which would silently break run-to-run
    reproducibility for a given base_seed.
    """
    if base_seed is None:
        return None
    return base_seed + (zlib.crc32(str(key).encode("utf-8")) % 100000)


# ---------------------------------------------------------------------------
# Visible grid watermark: per-tile jitter
# ---------------------------------------------------------------------------

def _build_grid_layer(
    size,
    text,
    font_size,
    opacity,
    angle,
    spacing,
    font_path,
    seed,
    jitter_pos,
    jitter_opacity,
    jitter_angle,
):
    """Build a transparent RGBA layer with the watermark repeated in a grid,
    where every tile gets its own random position offset, opacity, and
    rotation (all derived from `seed`, so the same seed always reproduces
    the exact same pattern).

    This is the plain, non-adaptive grid builder - kept as its own function
    (rather than inlined in create_watermark_layer) because adaptive
    placement needs to build *two* of these at different spacings (see
    create_watermark_layer below) and they must share this exact logic or
    the two grids could drift apart in look/feel.
    """
    rng = random.Random(seed)
    width, height = size
    diagonal = int(math.hypot(width, height)) + spacing * 2
    layer = Image.new("RGBA", (diagonal, diagonal), (0, 0, 0, 0))
    font = load_font(font_size, font_path)

    # Measure the text once so each tile canvas is sized correctly.
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = probe.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tile_size = int(math.hypot(tw, th)) + 24

    row = 0
    for y in range(0, diagonal, spacing):
        offset_x = (spacing // 2) if row % 2 else 0
        for x in range(-spacing, diagonal + spacing, spacing):
            dx = rng.uniform(-jitter_pos, jitter_pos) * spacing
            dy = rng.uniform(-jitter_pos, jitter_pos) * spacing
            tile_angle = angle + rng.uniform(-jitter_angle, jitter_angle)
            tile_opacity = max(0, min(255, opacity + rng.randint(-jitter_opacity, jitter_opacity)))

            tile = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
            tdraw = ImageDraw.Draw(tile)
            tdraw.text(
                ((tile_size - tw) // 2 - bbox[0], (tile_size - th) // 2 - bbox[1]),
                text,
                font=font,
                fill=(255, 255, 255, tile_opacity),
            )
            tile = tile.rotate(tile_angle, expand=True, resample=Image.BICUBIC)

            px = int(x + offset_x + dx - tile.width / 2)
            py = int(y + dy - tile.height / 2)
            layer.alpha_composite(tile, (px, py))
        row += 1

    left = (diagonal - width) // 2
    top = (diagonal - height) // 2
    return layer.crop((left, top, left + width, top + height))


def create_watermark_layer(
    size,
    text,
    font_size=36,
    opacity=90,
    angle=30,
    spacing=250,
    font_path=None,
    seed=None,
    jitter_pos=0.35,
    jitter_opacity=40,
    jitter_angle=8,
    sensitivity_map=None,
    adaptive_strength=0.6,
    infill_spacing_divisor=2.0,
):
    """Build the watermark layer for one image/frame - a plain jittered grid
    (see _build_grid_layer), optionally with an extra, denser grid layered
    on top wherever `sensitivity_map` says the removal is likely to be hard
    to hide (faces, high-detail "main subject" regions - see
    compute_sensitivity_map).

    Adaptive placement is implemented as a *second*, independently-jittered
    grid at a finer `spacing` (spacing / infill_spacing_divisor), masked by
    `sensitivity_map * adaptive_strength` and alpha-composited on top of the
    base grid. Two things fall out of that approach for free:
      - Outside the mask (sensitivity ~ 0), the infill layer's alpha is
        scaled to ~0, so the image is pixel-for-pixel identical to the
        non-adaptive path there - "off" really means off.
      - Because it's a *second* full grid rather than, say, inserting a few
        extra tiles by hand, it inherits the same per-tile jitter (and its
        own derived seed - see derive_seed) instead of adding a second,
        perfectly-regular pattern that would reintroduce the exact
        FFT-periodicity weakness per-tile jitter exists to break.

    `sensitivity_map`, if given, must be an (height, width) float array in
    0..1 matching `size`. When None (the default), this is exactly the old
    single-grid behavior - existing callers that never pass these new
    kwargs see no change at all.
    """
    base_layer = _build_grid_layer(
        size, text, font_size, opacity, angle, spacing, font_path, seed,
        jitter_pos, jitter_opacity, jitter_angle,
    )
    if sensitivity_map is None or adaptive_strength <= 0:
        return base_layer

    width, height = size
    mask = np.clip(sensitivity_map, 0, 1).astype(np.float32)
    if mask.shape != (height, width):
        raise ValueError(
            f"sensitivity_map shape {mask.shape} does not match image size "
            f"(height={height}, width={width}) - build it from an array of "
            "this same image/frame."
        )
    mask *= float(np.clip(adaptive_strength, 0, 1))

    infill_spacing = int(max(24, spacing / max(1.0, infill_spacing_divisor)))
    # A distinct derived seed, not the same seed reused: an identical
    # pattern stamped twice would just look like one bolder grid at the
    # *same* jitter, not two independent ones - and a shared seed here
    # would also make the infill grid's jitter perfectly correlated with
    # the base grid's, which is exactly the kind of predictable structure
    # per-tile jitter is meant to avoid introducing elsewhere in this file.
    infill_seed = derive_seed(seed, "adaptive-infill")
    infill_layer = _build_grid_layer(
        size, text, font_size, opacity, angle, infill_spacing, font_path, infill_seed,
        jitter_pos, jitter_opacity, jitter_angle,
    )

    infill_arr = np.array(infill_layer, dtype=np.float32)  # copy, not a view
    infill_arr[..., 3] *= mask
    infill_layer = Image.fromarray(np.clip(infill_arr, 0, 255).astype(np.uint8), "RGBA")

    return Image.alpha_composite(base_layer, infill_layer)


# ---------------------------------------------------------------------------
# Content-adaptive placement: detect "sensitive" regions (faces, high-detail
# main subject) so the grid above gets denser / darker exactly there
# ---------------------------------------------------------------------------

_face_cascade_cache = {}


def _get_face_cascade(cascade_path=None):
    """Load (and cache, keyed by path) a Haar cascade face detector.

    Haar cascades are the classic lightweight face model: a few hundred KB
    XML file, CPU-only, no GPU and no network download required - it ships
    inside opencv-python-headless itself (cv2.data.haarcascades), which
    this project already depends on for video frame decoding. That makes
    it a good fit here: real signal for the single most common "sensitive"
    region (someone's face is usually the most detail-rich, hardest area to
    inpaint over without a visible seam) at essentially no added cost or
    new trust surface over what the tool already needs.
    """
    key = cascade_path or "__default__"
    if key not in _face_cascade_cache:
        path = cascade_path or os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        clf = cv2.CascadeClassifier(path)
        if clf.empty():
            raise ValueError(f"Could not load face cascade from: {path}")
        _face_cascade_cache[key] = clf
    return _face_cascade_cache[key]


def compute_sensitivity_map(rgb_uint8, face_cascade_path=None, detail_weight=0.5, downscale=4, face_pad=0.4):
    """Build an (height, width) float32 map in 0..1 marking where a clean
    watermark-removal inpaint is most likely to leave visible artifacts:
    faces, and more generally any high-detail "main subject" area, as
    opposed to flat sky/wall/background-blur, which an inpainting model
    can regenerate almost invisibly.

    Two signals, both intentionally simple/fast rather than another neural
    net, combined with max() so either one alone is enough to flag a region:
      1. Face detection (Haar cascade - see _get_face_cascade). Specific
         and strong wherever it fires.
      2. Local detail energy: Laplacian magnitude, box-averaged over a
         neighborhood. A general "how textured/high-frequency is this
         patch" proxy - it also lights up on other detailed subjects
         (patterned clothing, foreground objects, on-image text) even with
         no face in frame, which plain face detection alone would miss.

    Both run on a downscaled copy for speed (`downscale`); the combined
    result is then Gaussian-blurred and resized back up, so the boost fades
    in smoothly around each detected region instead of showing a hard
    rectangular seam - a sharp-edged boost would itself be a visible tell
    that the image was processed differently there, working against the
    whole point.
    """
    h, w = rgb_uint8.shape[:2]
    small_w, small_h = max(1, w // downscale), max(1, h // downscale)
    small = cv2.resize(rgb_uint8, (small_w, small_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    # --- signal 1: faces --------------------------------------------------
    face_mask = np.zeros((small_h, small_w), dtype=np.float32)
    try:
        cascade = _get_face_cascade(face_cascade_path)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    except (cv2.error, ValueError):
        faces = ()  # a missing/corrupt custom --face-cascade: degrade to detail-only
    for (fx, fy, fw, fh) in faces:
        pad_x, pad_y = int(fw * face_pad), int(fh * face_pad)
        x0, y0 = max(0, fx - pad_x), max(0, fy - pad_y)
        x1, y1 = min(small_w, fx + fw + pad_x), min(small_h, fy + fh + pad_y)
        face_mask[y0:y1, x0:x1] = 1.0

    # --- signal 2: local detail energy -------------------------------------
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    detail = cv2.blur(np.abs(lap), (9, 9))
    # Robust-normalize: a handful of very sharp pixels (a hard edge, jpeg
    # blocking) shouldn't blow out the whole scale and wash out everything
    # else to near-1. Clip to the 95th percentile before scaling to 0..1.
    ceiling = np.percentile(detail, 95) or 1.0
    detail_norm = np.clip(detail / ceiling, 0, 1)

    combined = np.clip(np.maximum(face_mask, detail_norm * detail_weight), 0, 1)

    blur_ksize = max(3, (min(small_w, small_h) // 20) | 1)  # odd kernel size
    combined = cv2.GaussianBlur(combined, (blur_ksize, blur_ksize), 0)
    full = cv2.resize(combined, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(full, 0, 1).astype(np.float32)


def build_ink_map(ink, sensitivity_map, adaptive_strength, mode):
    """Turn a scalar --ink into a spatially-varying (height, width, 1) array
    when adaptive placement is on, so multiply/overlay blending reads
    darker/stronger exactly over `sensitivity_map`'s detected regions -
    without changing what --ink means anywhere the map is near 0.

    The direction of the shift always moves `ink` further from that blend
    mode's own no-op point (see the --ink docs on create_watermark_layer's
    callers): for multiply that's 1.0 (smaller ink = darker), for overlay
    it's 0.5 (moving toward either 0 or 1 intensifies, crossing 0.5 would
    flip which side of the no-op point you're on and invert the effect).
    Both branches below are written so the shift can approach but never
    reach - let alone cross - that no-op point, however large
    `adaptive_strength` is.

    blend_arrays()/composite() already do plain numpy arithmetic with
    `ink` (`base01 * ink`, etc.), so handing them an array here instead of
    a float works for free via numpy broadcasting - no change needed there.
    Returns `ink` unchanged for mode="alpha" (ink doesn't apply there) or
    when adaptive placement isn't in use, so existing callers are unaffected.
    """
    if sensitivity_map is None or adaptive_strength <= 0 or mode == "alpha":
        return ink
    s = np.clip(sensitivity_map, 0, 1).astype(np.float32) * float(np.clip(adaptive_strength, 0, 1))
    if mode == "multiply":
        ink_map = ink * (1.0 - s)
    elif mode == "overlay":
        if ink <= 0.5:
            ink_map = ink * (1.0 - s)
        else:
            ink_map = ink + s * (1.0 - ink)
    else:
        return ink
    return ink_map[..., None]


# ---------------------------------------------------------------------------
# Blend modes: mix the mark into local pixel values instead of flat alpha-over
# ---------------------------------------------------------------------------

def blend_arrays(base01, alpha01, mode="alpha", ink=0.32, mark_rgb01=None):
    """Pure-numpy blend core, shared by the still-image compositor and the
    per-frame video worker.

    base01:     HxWx3 float32 array, base pixels in 0..1.
    alpha01:    HxWx1 float32 array, watermark mask alpha in 0..1.
    mark_rgb01: HxWx3 float32 array, only needed for mode="alpha" (the
                watermark's own RGB to lay over the base). Not used for
                multiply/overlay, which derive the blended color from the
                base pixels themselves.
    Returns an HxWx3 float32 array in 0..1.
    """
    if mode == "alpha":
        if mark_rgb01 is None:
            raise ValueError('mode="alpha" requires mark_rgb01')
        return base01 * (1 - alpha01) + mark_rgb01 * alpha01

    if mode == "multiply":
        blended = base01 * ink
    elif mode == "overlay":
        blended = np.where(base01 < 0.5, 2 * base01 * ink, 1 - 2 * (1 - base01) * (1 - ink))
    else:
        raise ValueError(f"Unknown blend mode: {mode}")

    return base01 * (1 - alpha01) + blended * alpha01


def composite(base_rgba, mark_rgba, mode="alpha", ink=0.32):
    """Combine a still base image + watermark layer (both PIL RGBA images).

    mode="alpha":    standard alpha-over (original behavior).
    mode="multiply": darkens the base by `ink` wherever the mark's alpha
                      is set, scaled by that alpha - like ink soaking into
                      paper, so the result depends on the underlying pixel
                      value rather than sitting on a separable layer.
    mode="overlay":  classic overlay blend (darkens shadows, lightens
                      highlights less aggressively than multiply), also
                      scaled by the mark's alpha.
    """
    if mode == "alpha":
        return Image.alpha_composite(base_rgba, mark_rgba)

    base01 = np.asarray(base_rgba.convert("RGB"), dtype=np.float32) / 255.0
    alpha01 = np.asarray(mark_rgba.getchannel("A"), dtype=np.float32)[..., None] / 255.0
    out01 = blend_arrays(base01, alpha01, mode=mode, ink=ink)
    out_img = Image.fromarray(np.clip(out01 * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB")
    return out_img.convert("RGBA")


def watermark_layer_to_arrays(mark_rgba_image):
    """Convert a PIL RGBA watermark layer into the two float32 arrays the
    video frame workers need: RGB in 0..1 and alpha in 0..1 (kept separate
    so a worker never has to re-derive them per frame).
    """
    arr = np.asarray(mark_rgba_image, dtype=np.float32) / 255.0
    return np.ascontiguousarray(arr[..., :3]), np.ascontiguousarray(arr[..., 3:4])


# ---------------------------------------------------------------------------
# Invisible backup mark: simple LSB payload (not robust to recompression)
# ---------------------------------------------------------------------------

def _text_to_bits(text):
    payload = text.encode("utf-8")
    header = len(payload).to_bytes(4, "big")
    data = header + payload
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def embed_invisible(base_rgba, text):
    """Spread `text` across the low bits of the R channel, repeated to fill
    the image (so it survives a crop that keeps a reasonably large region).
    Lossless output only (save as PNG) - JPEG recompression destroys this.
    """
    arr = np.asarray(base_rgba.convert("RGB"), dtype=np.uint8).copy()
    flat = arr[..., 0].reshape(-1)  # red channel, flattened
    bits = _text_to_bits(text)
    if not bits:
        return base_rgba
    reps = max(1, len(flat) // len(bits))
    tiled_bits = (bits * (reps + 1))[: len(flat)]
    bit_arr = np.array(tiled_bits, dtype=np.uint8)
    flat[:] = (flat & 0xFE) | bit_arr
    arr[..., 0] = flat.reshape(arr[..., 0].shape)
    out = Image.fromarray(arr, "RGB")
    return out.convert("RGBA")


# ---------------------------------------------------------------------------
# Provenance metadata
# ---------------------------------------------------------------------------

def save_with_metadata(image, output_path, author=None, copyright_text=None, jpeg_quality=95):
    is_jpeg = output_path.lower().endswith((".jpg", ".jpeg"))

    if is_jpeg:
        rgb = image.convert("RGB")
        exif_bytes = None
        if (author or copyright_text) and piexif is not None:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
            if author:
                exif_dict["0th"][piexif.ImageIFD.Artist] = author.encode("utf-8")
            if copyright_text:
                exif_dict["0th"][piexif.ImageIFD.Copyright] = copyright_text.encode("utf-8")
            exif_bytes = piexif.dump(exif_dict)
        elif (author or copyright_text) and piexif is None:
            print("Note: install piexif (pip install piexif) to embed EXIF author/copyright in JPEGs.")
        if exif_bytes:
            rgb.save(output_path, quality=jpeg_quality, exif=exif_bytes)
        else:
            rgb.save(output_path, quality=jpeg_quality)
    else:
        pnginfo = None
        if author or copyright_text:
            pnginfo = PngImagePlugin.PngInfo()
            if author:
                pnginfo.add_text("Author", author)
            if copyright_text:
                pnginfo.add_text("Copyright", copyright_text)
        if pnginfo:
            image.save(output_path, pnginfo=pnginfo)
        else:
            image.save(output_path)
