"""Image sources for the benchmark.

The unit tests use one synthetic image (gradient + white noise). That is fine
for pass/fail, but a benchmark that reports a *rate* needs many images and,
ideally, images whose DCT statistics resemble photos - a mark that survives
JPEG q=30 on white-noise texture says little about a smooth sky.

Three sources, all described by a small JSON-able dict (`spec`) so a worker
process can rebuild the exact image from the seed instead of pickling pixels:

  {"kind": "gradnoise", "w", "h", "seed"}   the tests' image: gradient + N(0, 8) noise.
  {"kind": "pink",      "w", "h", "seed"}   1/f-spectrum texture + smooth colour field +
                                            hard-edged shapes + mild sensor noise. Much
                                            closer to a natural image's power spectrum,
                                            and harder: flat regions and sharp edges
                                            carry very different mid-band DCT energy.
  {"kind": "file",      "path", "max_side"} a real image from --images DIR.

Synthetic images are a stand-in, not a substitute: numbers published for the
tool should come from a real corpus (--images), and every report says which
image kinds it was measured on.
"""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def _gradnoise(w, h, seed):
    xx, yy = np.meshgrid(np.linspace(0, 255, w), np.linspace(0, 255, h))
    rng = np.random.default_rng(seed)
    arr = np.stack([xx, yy, (xx + yy) / 2], axis=-1)
    arr = np.clip(arr + rng.normal(0, 8, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _pink_field(rng, h, w, beta=1.0):
    """Unit-variance noise whose amplitude spectrum falls as 1/f**beta
    (beta ~ 1 is typical of natural images)."""
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = 1.0
    spec = (rng.normal(size=f.shape) + 1j * rng.normal(size=f.shape)) / np.power(f, beta)
    spec[0, 0] = 0.0
    field = np.fft.irfft2(spec, s=(h, w))
    return (field - field.mean()) / (field.std() + 1e-9)


def _pink(w, h, seed):
    rng = np.random.default_rng(seed)
    beta = rng.uniform(0.8, 1.4)          # per-image spectral slope
    texture_amp = rng.uniform(12, 40)     # how "busy" the image is
    # Smooth colour field: pink noise at 1/4 resolution, upsampled.
    hs, ws = max(2, h // 4), max(2, w // 4)
    luma = _pink_field(rng, hs, ws, beta=1.6)[..., None]
    small = luma * 35 + np.stack([_pink_field(rng, hs, ws, beta=1.6) for _ in range(3)], axis=-1) * 12
    colour = np.asarray(Image.fromarray((small + 128).clip(0, 255).astype(np.uint8), "RGB")
                        .resize((w, h), Image.BICUBIC), dtype=np.float64)
    # Luma texture shared by all channels (+ small independent chroma jitter).
    tex = _pink_field(rng, h, w, beta)[..., None] * texture_amp
    chroma = np.stack([_pink_field(rng, h, w, beta + 0.5) for _ in range(3)], axis=-1) * (texture_amp * 0.25)
    arr = colour + tex + chroma

    # Hard-edged shapes (flat fill, slightly feathered): edges are where real
    # images concentrate high-frequency DCT energy and where JPEG hurts most.
    layer = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(layer)
    mask = Image.new("L", (w, h), 0)
    mdraw = ImageDraw.Draw(mask)
    for _ in range(int(rng.integers(4, 10))):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x1, y1 = x0 + int(rng.integers(w // 12, w // 3)), y0 + int(rng.integers(h // 12, h // 3))
        col = tuple(int(c) for c in rng.integers(20, 236, 3))
        if rng.random() < 0.5:
            draw.ellipse([x0, y0, x1, y1], fill=col)
            mdraw.ellipse([x0, y0, x1, y1], fill=255)
        else:
            draw.rectangle([x0, y0, x1, y1], fill=col)
            mdraw.rectangle([x0, y0, x1, y1], fill=255)
    mask = np.asarray(mask.filter(ImageFilter.GaussianBlur(0.8)), dtype=np.float64)[..., None] / 255.0
    arr = arr * (1 - mask) + np.asarray(layer, dtype=np.float64) * mask
    arr += rng.normal(0, 1.5, arr.shape)  # sensor-ish noise
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


_SYNTH = {"gradnoise": _gradnoise, "pink": _pink}


def load_source(spec):
    """Rebuild an image (RGBA, as watermark.py's pipeline hands to the embedders)
    from its spec."""
    kind = spec["kind"]
    if kind in _SYNTH:
        img = _SYNTH[kind](spec["w"], spec["h"], spec["seed"])
    elif kind == "file":
        img = Image.open(spec["path"]).convert("RGB")
        m = spec.get("max_side")
        if m and max(img.size) > m:
            s = m / max(img.size)
            img = img.resize((max(8, round(img.width * s)), max(8, round(img.height * s))), Image.LANCZOS)
    else:
        raise ValueError(f"unknown image source kind: {kind!r}")
    return img.convert("RGBA")


def source_id(spec):
    if spec["kind"] == "file":
        return os.path.basename(spec["path"])
    return f"{spec['kind']}-{spec['w']}x{spec['h']}-s{spec['seed']}"


def list_real_images(directory, limit=None):
    """Sorted (so runs are reproducible) image files under `directory`."""
    paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                paths.append(os.path.join(root, f))
    paths.sort()
    return paths[:limit] if limit else paths


def make_specs(kinds, sizes, n, seed0, images_dir=None, max_side=1600):
    """`n` specs per (kind, size) for synthetic kinds, or up to `n` real files
    (kind "file"; sizes are ignored - a real image keeps its own size)."""
    specs = []
    if images_dir:
        for p in list_real_images(images_dir, n):
            specs.append({"kind": "file", "path": p, "max_side": max_side})
        return specs
    for kind in kinds:
        for (w, h) in sizes:
            for i in range(n):
                specs.append({"kind": kind, "w": w, "h": h, "seed": seed0 + i})
    return specs
