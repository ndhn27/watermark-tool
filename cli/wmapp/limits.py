"""Resource limits. The CLI and the Python API run your own code on your own
files, so they default to NO_LIMITS. The HTTP adapter takes input from the
network and passes SERVER_LIMITS: without them a request like spacing=1 on a
large image asks for hundreds of millions of tile draws and pins a core for
hours - not a bug in the grid code, just an input nobody should be able to
send. Limits are enforced inside the pipeline (not only in the adapter) so
they hold for every caller that opts in."""

import math
from dataclasses import dataclass
from typing import Optional

from .errors import LimitError


@dataclass(frozen=True)
class Limits:
    max_pixels: Optional[int] = None
    max_tiles: Optional[int] = None
    max_text_chars: Optional[int] = None
    max_upload_bytes: Optional[int] = None       # checked by adapters, which see the bytes first

    def check_text(self, text):
        if self.max_text_chars is not None and text is not None and len(text) > self.max_text_chars:
            raise LimitError(f"text is {len(text)} characters; this interface allows at most {self.max_text_chars}")

    def check_image(self, size, spec):
        w, h = size
        if self.max_pixels is not None and w * h > self.max_pixels:
            raise LimitError(f"image is {w}x{h} ({w * h / 1e6:.1f} MP); this interface allows at most "
                             f"{self.max_pixels / 1e6:.0f} MP")
        if self.max_tiles is not None:
            tiles = estimate_tiles(size, spec.spacing, spec.adaptive, spec.adaptive_infill_divisor)
            if tiles > self.max_tiles:
                raise LimitError(f"spacing={spec.spacing} on a {w}x{h} image means about {tiles:,} watermark tiles; "
                                 f"this interface allows at most {self.max_tiles:,} - use a larger spacing")


def estimate_tiles(size, spacing, adaptive=False, infill_divisor=2.0):
    """How many tiles wmcore's grid builder will draw for an image of `size`
    (mirrors the loops in wmcore.visible._build_grid_layer; the adaptive infill
    grid is denser by infill_divisor in each direction)."""
    w, h = size

    def one(sp):
        diagonal = int(math.hypot(w, h)) + sp * 2
        return math.ceil(diagonal / sp) * math.ceil((diagonal + 2 * sp) / sp)

    total = one(spacing)
    if adaptive:      # wmcore.create_watermark_layer: infill_spacing = int(max(24, spacing / max(1, divisor)))
        total += one(int(max(24, spacing / max(1.0, infill_divisor))))
    return total


NO_LIMITS = Limits()
SERVER_LIMITS = Limits(max_pixels=40_000_000, max_tiles=60_000, max_text_chars=200,
                       max_upload_bytes=30 * 1024 * 1024)
