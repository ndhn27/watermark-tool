"""wmcore - shared watermarking primitives used by watermark.py (images),
video.py (video frames), and their multiprocessing workers.

Split by concern into submodules (primitives, blend, visible, adaptive,
invisible, spread, rescale, metadata, wmid); this file re-exports their public
API as a flat `wmcore.*` namespace so callers keep writing
`import wmcore; wmcore.create_watermark_layer(...)` unchanged.

Two constraints shape how code is organized across these submodules:
  - It's the single source of truth for the watermark math, so the image
    and video paths can't silently drift apart.
  - Anything handed to multiprocessing.Pool / ProcessPoolExecutor under the
    "spawn" start method (default on Windows/macOS) must be picklable by
    reference - a plain top-level (sub)module import, not a closure or
    something defined in a __main__ script. Every function here qualifies,
    since wmcore and its submodules are all regularly importable.
"""

from . import adaptive, blend, invisible, metadata, orient, primitives, rescale, resync, spread, visible, wmid
from .adaptive import AdaptiveTracker, _get_face_cascade, build_ink_map, compute_sensitivity_map
from .blend import blend_arrays, composite, watermark_layer_to_arrays
from .invisible import embed_invisible
from .orient import (
    ORIENTATION_LABELS,
    ORIENTATIONS,
    OrientedMatch,
    find_spread_spectrum_id_oriented,
    find_spread_spectrum_payload_or_id_oriented,
)
from .metadata import save_with_metadata
from .primitives import (
    DEFAULT_FONT_PATHS,
    IMAGE_EXTS,
    VIDEO_EXTS,
    derive_seed,
    load_font,
    text_to_bits,
)
from .resync import (
    DEFAULT_RESYNC_BUDGET_SECONDS,
    ResyncMatch,
    ScaledMatch,
    find_spread_spectrum_id_resync,
    find_spread_spectrum_id_resync_scaled,
)
from .rescale import (
    DEFAULT_RESCALE_BUDGET_SECONDS,
    SS_RESCALE_RANGE,
    RescaleMatch,
    _printable_payload,
    _ss_patch_bimodality,
    _ss_refine_scale,
    _ss_sweep_peaks,
    _SS_REFINE_STAGES,
    _SS_SWEEP_TIERS,
    find_spread_spectrum_payload_rescaled,
)
from .spread import (
    DEFAULT_SS_ALPHA,
    DEFAULT_SS_ALPHA_VIDEO,
    SS_BLOCK,
    SS_COEFF_POSITIONS,
    _dct_basis_matrix,
    _fmix32,
    _image_to_ss_blocks,
    _rgb_to_y,
    _ss_blocks_to_image,
    _ss_correlations_from_luma,
    _ss_pn_vector,
    _ycbcr_to_rgb,
    embed_spread_spectrum,
    ID_PERIOD_BITS,
    find_spread_spectrum_id,
    find_spread_spectrum_payload,
    find_spread_spectrum_payload_or_id,
    spread_spectrum_block_correlations,
    spread_spectrum_block_correlations_multi,
)
from .visible import _build_grid_layer, create_watermark_layer

# Old private name, kept as an alias - see primitives.text_to_bits.
_text_to_bits = text_to_bits

__all__ = [
    "adaptive", "blend", "invisible", "metadata", "orient", "primitives", "rescale", "resync", "spread", "visible", "wmid",
    "DEFAULT_RESYNC_BUDGET_SECONDS", "ResyncMatch", "ScaledMatch", "find_spread_spectrum_id_resync",
    "find_spread_spectrum_id_resync_scaled",
    "DEFAULT_FONT_PATHS", "VIDEO_EXTS", "IMAGE_EXTS", "load_font", "derive_seed", "text_to_bits",
    "blend_arrays", "composite", "watermark_layer_to_arrays",
    "create_watermark_layer",
    "compute_sensitivity_map", "build_ink_map", "AdaptiveTracker",
    "embed_invisible",
    "embed_spread_spectrum", "spread_spectrum_block_correlations", "spread_spectrum_block_correlations_multi",
    "find_spread_spectrum_payload", "find_spread_spectrum_id", "find_spread_spectrum_payload_or_id", "ID_PERIOD_BITS",
    "DEFAULT_SS_ALPHA", "DEFAULT_SS_ALPHA_VIDEO", "SS_BLOCK", "SS_COEFF_POSITIONS",
    "find_spread_spectrum_payload_rescaled", "SS_RESCALE_RANGE", "RescaleMatch",
    "DEFAULT_RESCALE_BUDGET_SECONDS",
    "ORIENTATIONS", "ORIENTATION_LABELS", "OrientedMatch",
    "find_spread_spectrum_id_oriented", "find_spread_spectrum_payload_or_id_oriented",
    "save_with_metadata",
]
