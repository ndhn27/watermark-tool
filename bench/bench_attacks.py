"""Attack registry: the transformations a marked image is put through before
decoding. Each attack is `fn(rgb_image, rng) -> (rgb_image, Geometry)`.

`Geometry` tells the benchmark how the attacked image maps back onto the
original block grid, which is what lets it compute an ORACLE bit error rate
(signal remaining, independent of whether the decoder can re-synchronise):

  identity     same pixel grid (JPEG, noise, blur, tone changes...).
  resize       uniform rescale by `scale`; the oracle resamples back to the
               original size exactly.
  crop_blocks  crop by a whole number of 8x8 blocks (cy rows, cx cols) from
               the top/left; the oracle knows which original block each
               remaining block was.
  flip         mirrored horizontally / vertically / both (`flip` = "h" | "v" |
               "hv"), optionally followed by a JPEG re-save; un-mirroring the
               attacked image restores the original block grid exactly, so
               the oracle is that un-mirror.
  none         no oracle mapping exists (unaligned crop, rotation) - BER is
               reported as n/a.

`gate=False` marks attacks the README documents as NOT supported (or only
partly): they are measured and printed, flagged with a dagger, and excluded
from the regression gate - a "known limit" moving is information, not a
failure. (run_benchmark.OUT_OF_SCOPE adds a per-decoder layer on top: e.g. a
resize is in scope for `ss_rescale` but not for plain `ss`.)
"""

import io
from typing import Callable, NamedTuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageFilter, features


class Geometry(NamedTuple):
    kind: str                                   # identity | resize | crop_blocks | flip | none
    scale: Optional[float] = None               # resize: attacked_size / original_size
    crop_blocks: Optional[tuple] = None         # crop_blocks: (rows, cols) removed from top-left
    flip: Optional[str] = None                  # flip: "h" | "v" | "hv"


class Attack(NamedTuple):
    name: str
    family: str
    fn: Callable
    gate: bool = True
    note: str = ""


REGISTRY = {}


def _register(name, family, fn, gate=True, note=""):
    REGISTRY[name] = Attack(name, family, fn, gate, note)


def _ident(img, rng):
    return img, Geometry("identity")


def _jpeg_bytes(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _jpeg(quality):
    def fn(img, rng):
        return _jpeg_bytes(img, quality), Geometry("identity")
    return fn


def _resize(scale, jpeg_quality=None, resample=Image.BICUBIC):
    def fn(img, rng):
        w, h = img.size
        out = img.resize((max(8, round(w * scale)), max(8, round(h * scale))), resample)
        if jpeg_quality is not None:
            out = _jpeg_bytes(out, jpeg_quality)
        return out, Geometry("resize", scale=scale)
    return fn


def _crop_blocks(px_top, px_left):
    assert px_top % 8 == 0 and px_left % 8 == 0
    def fn(img, rng):
        arr = np.asarray(img)
        out = Image.fromarray(np.ascontiguousarray(arr[px_top:, px_left:]), "RGB")
        return out, Geometry("crop_blocks", crop_blocks=(px_top // 8, px_left // 8))
    return fn


def _crop_unaligned(px):
    def fn(img, rng):
        arr = np.asarray(img)
        return Image.fromarray(np.ascontiguousarray(arr[px:, px:]), "RGB"), Geometry("none")
    return fn


def _noise(sigma):
    def fn(img, rng):
        arr = np.asarray(img, dtype=np.float64)
        arr = arr + rng.normal(0.0, sigma, arr.shape)
        return Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8), "RGB"), Geometry("identity")
    return fn


def _blur(radius):
    def fn(img, rng):
        return img.filter(ImageFilter.GaussianBlur(radius)), Geometry("identity")
    return fn


def _tone(brightness=0.0, contrast=1.0, gamma=1.0):
    def fn(img, rng):
        a = np.asarray(img, dtype=np.float64) / 255.0
        a = np.power(np.clip(a, 0, 1), gamma)
        a = (a - 0.5) * contrast + 0.5 + brightness / 255.0
        return Image.fromarray(np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB"), Geometry("identity")
    return fn


def _rotate(deg):
    def fn(img, rng):
        return img.rotate(deg, resample=Image.BICUBIC), Geometry("none")
    return fn


def _webp(quality):
    def fn(img, rng):
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB"), Geometry("identity")
    return fn


def _avif(quality):
    def fn(img, rng):
        buf = io.BytesIO()
        img.save(buf, format="AVIF", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB"), Geometry("identity")
    return fn


def _flip_then_resize(kind, scale):
    """Mirror, then resize (bicubic) - a repost that mirrors the picture and rescales it. No oracle geometry:
    an oracle would have to compose the two (the rescale search's own job), so BER is n/a."""
    flip, resize = _flip(kind), _resize(scale)
    def fn(img, rng):
        out, _ = flip(img, rng)
        out, _ = resize(out, rng)
        return out, Geometry("none")
    return fn


def _flip_horizontal():
    def fn(img, rng):
        return img.transpose(Image.FLIP_LEFT_RIGHT), Geometry("none")
    return fn


def _flip(kind, jpeg_quality=None):
    """Mirror the image ("h" = left-right, "v" = top-bottom, "hv" = both, i.e. a 180-degree turn), then
    optionally re-save as JPEG (what a repost that mirrors the picture and re-encodes it does). Carries an
    oracle geometry (`flip`), unlike the older `flip_horizontal` above, which is kept as it was so P0's
    records stay comparable."""
    def fn(img, rng):
        out = img
        if "h" in kind:
            out = out.transpose(Image.FLIP_LEFT_RIGHT)
        if "v" in kind:
            out = out.transpose(Image.FLIP_TOP_BOTTOM)
        if jpeg_quality is not None:
            out = _jpeg_bytes(out, jpeg_quality)
        return out, Geometry("flip", flip=kind)
    return fn


def _perspective(strength):
    """Nudge each of the 4 corners inward/outward by up to `strength` (as a
    fraction of width/height) and warp to match - a stand-in for an
    off-axis photo of a photo / screen recapture, not a perfectly
    fronto-parallel copy. No oracle geometry: a homography isn't a case
    bench_schemes.oracle_soft knows how to invert, so BER is n/a here the
    same way it already is for `rotate_*deg`.
    """
    def fn(img, rng):
        w, h = img.size
        dx, dy = w * strength, h * strength
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = (src + rng.uniform(-1, 1, size=(4, 2)) * [dx, dy]).astype(np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        arr = np.asarray(img.convert("RGB"))
        out = cv2.warpPerspective(arr, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(out, "RGB"), Geometry("none")
    return fn


def _pipeline(*steps):
    """Chain attack functions in sequence, simulating a realistic repost
    pipeline (resize + recompress + crop, all in one pass) instead of one
    isolated transform. Deliberately does not try to compose an oracle
    geometry across steps - the whole point of a pipeline attack is
    exercising a compound transform the single-step oracle mapping doesn't
    model - so BER is always n/a here (Geometry("none")); detection rate
    and FPR are what these attacks are actually for.
    """
    def fn(img, rng):
        cur = img
        for step in steps:
            cur, _ = step(cur, rng)
        return cur, Geometry("none")
    return fn


_register("identity", "identity", _ident)
for _q in (95, 85, 75, 60, 50, 30):
    _register(f"jpeg_{_q}", "jpeg", _jpeg(_q), gate=(_q >= 50),
              note="" if _q >= 50 else "README tests clean down to q=50 only")
if features.check("webp"):
    _register("webp_80", "jpeg", _webp(80))
if features.check("avif"):
    # Gated as of 2026-09-21: 24/24 detected, 0 misdecode at n=12/image-kind
    # (bench/results/p0-new-attacks-n12/), same shape as jpeg_75/webp_80 on
    # this synthetic corpus. Still not confirmed on real photos - see
    # bench/README.md's Attacks section for the caveat.
    _register("avif_60", "jpeg", _avif(60))
for _s in (0.9, 0.75, 0.62, 1.25, 1.5):
    _register(f"resize_{_s}", "resize", _resize(_s))
_register("resize_0.5", "resize", _resize(0.5), gate=False, note="README: ~0.5x can erase the mark")
_register("resize_0.75_jpeg85", "resize_jpeg", _resize(0.75, 85))
_register("resize_1.4_jpeg85", "resize_jpeg", _resize(1.4, 85))
_register("crop_top_64", "crop", _crop_blocks(64, 0))
_register("crop_topleft_64", "crop", _crop_blocks(64, 64), gate=False,
          note="left crop changes the block-column count, scrambling the raster tiling")
_register("crop_unaligned_3", "crop", _crop_unaligned(3), gate=False, note="README: block-aligned crops only")
for _px in (1, 7, 15):
    _register(f"crop_unaligned_{_px}", "crop", _crop_unaligned(_px), gate=False,
              note="README: block-aligned crops only")
for _s in (2, 5, 10):
    _register(f"noise_{_s}", "noise", _noise(_s), gate=(_s <= 5))
for _r in (0.5, 1.0, 1.5):
    _register(f"blur_{_r}", "blur", _blur(_r), gate=(_r <= 1.0))
_register("brightness_+25", "tone", _tone(brightness=25))
_register("contrast_0.8", "tone", _tone(contrast=0.8))
_register("gamma_1.3", "tone", _tone(gamma=1.3))
for _deg in (1.0, 2.0, 3.0, 5.0):
    _register(f"rotate_{_deg:g}deg", "geometric", _rotate(_deg), gate=False,
              note="README: desynchronisation not handled")
_register("flip_horizontal", "geometric", _flip_horizontal(), gate=False,
          note="mirrors the block raster order; not handled")
# Family "flip": mirrored copies (optionally re-saved as JPEG). In scope only for the mirror-aware decoders
# (ss_flip, ss_id_flip - see run_benchmark.OUT_OF_SCOPE); the older flip_horizontal above stays in "geometric".
_register("flip_vertical", "flip", _flip("v"), note="top-bottom mirror; read only by the mirror-aware decoders")
_register("flip_both", "flip", _flip("hv"), note="mirrored both ways = 180-degree turn; read only by the mirror-aware decoders")
_register("flip_horizontal_jpeg_75", "flip", _flip("h", 75), note="left-right mirror + JPEG q75 re-save")
_register("flip_vertical_jpeg_50", "flip", _flip("v", 50), note="top-bottom mirror + JPEG q50 re-save")
_register("flip_horizontal_resize_0.75", "pipeline", _flip_then_resize("h", 0.75), gate=False,
          note="mirror + resize x0.75; not handled by plain decoders (BER n/a)")
_register("flip_vertical_resize_1.25", "pipeline", _flip_then_resize("v", 1.25), gate=False,
          note="mirror + resize x1.25; not handled by plain decoders (BER n/a)")
_register("perspective_light", "geometric", _perspective(0.03), gate=False,
          note="README: homography desync not handled")
_register("perspective_medium", "geometric", _perspective(0.06), gate=False,
          note="README: homography desync not handled")
_register("pipeline_resize_jpeg_crop", "pipeline",
          _pipeline(_resize(0.75), _jpeg(70), _crop_unaligned(5)),
          gate=False, note="simulated repost pipeline (resize .75x + jpeg q70 + 5px crop) - BER n/a, compound transform")
_register("pipeline_heavy_repost", "pipeline",
          _pipeline(_resize(0.6), _jpeg(65), _rotate(1.0), _crop_unaligned(4)),
          gate=False, note="simulated worse-case repost pipeline - BER n/a, compound transform")

QUICK_ATTACKS = ("identity", "jpeg_75", "jpeg_50", "resize_0.75", "resize_1.25", "crop_top_64", "noise_5", "blur_1.0")

# Negatives (no mark present) are run through a small representative subset -
# the point is FPR under the transformations a decoder will actually meet,
# especially resizes, where the rescale search tries many candidates and so
# has the most opportunities to hallucinate a payload.
NEGATIVE_ATTACKS = ("identity", "jpeg_75", "resize_0.75", "resize_1.25")


def select(patterns):
    """Attack names matching any of `patterns` (exact name or family; '*'
    globs allowed), in registry order."""
    import fnmatch
    if not patterns:
        return list(REGISTRY)
    out = []
    for name, atk in REGISTRY.items():
        if any(fnmatch.fnmatch(name, p) or atk.family == p for p in patterns):
            out.append(name)
    return out
