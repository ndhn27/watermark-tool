"""Content-adaptive placement: detect regions where a clean watermark-removal
inpaint is most likely to leave visible artifacts (faces, high-detail main
subjects) so visible.create_watermark_layer() can go denser/darker exactly
there.

For video, a single detection pass on frame 0 (the only option before
AdaptiveTracker below existed) goes stale the moment the subject or the
camera moves - see video.py's module docstring for the "detect once, reuse
forever" limitation this was built to replace.
"""

import os

import cv2
import numpy as np

_face_cascade_cache = {}


def _get_face_cascade(cascade_path=None):
    """Load (and cache, keyed by path) a Haar cascade face detector - a
    lightweight, CPU-only, no-download model that ships inside
    opencv-python-headless, which this project already depends on.
    """
    key = cascade_path or "__default__"
    if key not in _face_cascade_cache:
        path = cascade_path or os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        clf = cv2.CascadeClassifier(path)
        if clf.empty():
            raise ValueError(f"Could not load face cascade from: {path}")
        _face_cascade_cache[key] = clf
    return _face_cascade_cache[key]


def _sensitivity_from_small_gray(gray, face_cascade_path, detail_weight, face_pad):
    """The actual detection work, on an already-downscaled grayscale image.
    Returns the combined, blurred map at `gray`'s own (small) resolution -
    NOT yet resized up to the source image size. Split out of
    compute_sensitivity_map() so AdaptiveTracker (below) can reuse it without
    paying for a second downscale of the full-resolution frame every time it
    re-detects.
    """
    small_h, small_w = gray.shape[:2]

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
    # Robust-normalize against the 95th percentile so a handful of very sharp
    # pixels (a hard edge, jpeg blocking) doesn't wash out the whole scale.
    ceiling = np.percentile(detail, 95) or 1.0
    detail_norm = np.clip(detail / ceiling, 0, 1)

    combined = np.clip(np.maximum(face_mask, detail_norm * detail_weight), 0, 1)

    blur_ksize = max(3, (min(small_w, small_h) // 20) | 1)  # odd kernel size
    return cv2.GaussianBlur(combined, (blur_ksize, blur_ksize), 0)


def compute_sensitivity_map(rgb_uint8, face_cascade_path=None, detail_weight=0.5, downscale=4, face_pad=0.4):
    """Build an (height, width) float32 map in 0..1 marking where removal is
    hardest to hide invisibly: faces, and more generally any high-detail
    "main subject" area (as opposed to flat sky/wall/background-blur, which
    an inpainting model can regenerate almost invisibly).

    Two fast signals combined with max() so either alone is enough to flag
    a region: face detection (specific and strong wherever it fires) and
    local Laplacian detail energy (a general texture proxy that also catches
    patterned clothing, foreground objects, on-image text). Both run on a
    downscaled copy for speed (see _sensitivity_from_small_gray), then the
    combined map is resized back up so the boost fades in smoothly instead
    of showing a hard rectangular seam.
    """
    h, w = rgb_uint8.shape[:2]
    small_w, small_h = max(1, w // downscale), max(1, h // downscale)
    small = cv2.resize(rgb_uint8, (small_w, small_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    combined = _sensitivity_from_small_gray(gray, face_cascade_path, detail_weight, face_pad)
    full = cv2.resize(combined, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(full, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Temporal tracking (video): carry a detected sensitivity map forward across
# frames instead of freezing it at whatever frame it was first computed on.
# ---------------------------------------------------------------------------

def _dense_flow(prev_gray_small, curr_gray_small):
    """Farneback dense optical flow between two same-shape grayscale images.
    Deliberately the plain algorithm built into opencv-python-headless (no
    opencv-contrib) - this project's one cv2 dependency (see requirements.txt)
    already relies on the base package's Haar cascades staying dependency-free,
    and a fancier tracker (CSRT/KCF/MOSSE) would pull in opencv-contrib-python
    just for this.
    """
    return cv2.calcOpticalFlowFarneback(
        prev_gray_small, curr_gray_small, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )


def _warp_with_flow(prev_map_small, flow):
    """Advance `prev_map_small` (indexed in the PREVIOUS frame's pixel grid)
    to an estimate indexed in the CURRENT frame's grid, using dense flow
    from _dense_flow(prev, curr) - i.e. flow[y, x] is how far the prev
    pixel at (x, y) moved to land in curr (cv2's own convention: prev(x, y)
    corresponds to curr(x + flow_x(x, y), y + flow_y(x, y))).

    cv2.remap does backward sampling - dst(x, y) = src(mapx(x, y), mapy(x,
    y)) - so producing the current-frame estimate means, for each output
    position, looking up where in prev that content came FROM, not where
    it's going. Under the (reasonable, for a pan or slow motion) assumption
    that flow varies slowly enough for its value at a destination pixel to
    stand in for its value at that pixel's true source, the source is
    approximately `output_position - flow(output_position)` - subtraction,
    not addition. Getting this backwards would sample twice as far as
    intended in the direction of motion instead of undoing it.

    Pixels flowing in from outside the frame (newly revealed background on
    a pan) fall back to BORDER_REPLICATE - an edge guess, not a hole."""
    small_h, small_w = prev_map_small.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(small_w, dtype=np.float32), np.arange(small_h, dtype=np.float32))
    remap_x = grid_x - flow[..., 0]
    remap_y = grid_y - flow[..., 1]
    return cv2.remap(
        prev_map_small, remap_x, remap_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


class AdaptiveTracker:
    """Keeps a sensitivity map (see compute_sensitivity_map) current across
    an entire video instead of computing it once from the first frame and
    reusing that snapshot for every later frame regardless of what moves.

    Call update(frame_bgr) once per decoded frame, in order; it returns the
    current best (height, width) float32 map for that frame. Internally,
    each call does one of two things:

      - MOST frames: warp the last known map forward with dense optical flow
        between the previous and current (downscaled) grayscale frame. This
        is cheap enough to run every single frame in the main decode process
        (a handful of milliseconds on a small image) and is what lets the
        boosted region follow a moving subject or a panning/zooming camera
        instead of staying frozen wherever it was first detected.
      - Periodically (every `redetect_every_frames` frames), or whenever a
        scene cut is flagged, or on the very first call: re-run the actual
        face/detail detection (compute_sensitivity_map's math) and merge it
        with the flow-warped estimate via max() - the same "either signal is
        enough" combination compute_sensitivity_map already uses for its own
        two signals. This corrects the drift optical flow accumulates over
        time and picks up new subjects entering frame; flow alone would
        never notice a second face that walks in from off-camera.

    A scene cut (see `last_cut` and cut_threshold) skips the flow-warp
    entirely and detects fresh instead: flow computed across a hard content
    change describes noise, not motion, and warping the old map through it
    would drag stale detections into the new shot.

    This class only tracks a *sensitivity map* over time; it is deliberately
    unaware of the mark's own jitter-refresh segments (video.py's
    `frames_per_segment`) - those are a separate, coarser cadence for a
    different purpose (breaking up averaging-attack windows), driven by
    time, not content. video.py reads whatever `update()` last returned at
    each segment boundary to build that segment's mark/ink arrays; anything
    "temporal" video.py itself supports (like a scene cut also forcing an
    early segment boundary) is video.py's decision, not this class's.
    """

    def __init__(
        self,
        face_cascade_path=None,
        detail_weight=0.5,
        downscale=4,
        face_pad=0.4,
        redetect_every_frames=30,
        cut_threshold=0.12,
    ):
        self.face_cascade_path = face_cascade_path
        self.detail_weight = detail_weight
        self.downscale = max(1, int(downscale))
        self.face_pad = face_pad
        self.redetect_every_frames = max(1, int(redetect_every_frames))
        self.cut_threshold = cut_threshold

        self._map_small = None   # last known combined map, at small resolution
        self._gray_small = None  # last frame's small grayscale (flow + cut detection)
        self._frames_since_detect = 0
        self.last_cut = False    # exposed so callers can react too (e.g. force a segment boundary)

    def update(self, rgb_uint8):
        h, w = rgb_uint8.shape[:2]
        small_w, small_h = max(1, w // self.downscale), max(1, h // self.downscale)
        small = cv2.resize(rgb_uint8, (small_w, small_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

        same_shape = self._gray_small is not None and self._gray_small.shape == gray.shape
        cut = False
        if same_shape:
            # Mean absolute grayscale change, normalized to 0..1 - a cheap,
            # order-of-magnitude proxy for "did the shot just change", not a
            # real shot-boundary-detection algorithm. Good enough here: a
            # false positive just triggers one extra (cheap) re-detection;
            # a missed cut just falls back to the periodic redetect cadence.
            cut = float(np.mean(cv2.absdiff(gray, self._gray_small))) / 255.0 > self.cut_threshold
        self.last_cut = cut

        tracked_small = None
        if self._map_small is not None and same_shape and not cut:
            flow = _dense_flow(self._gray_small, gray)
            tracked_small = np.clip(_warp_with_flow(self._map_small, flow), 0, 1)

        need_detect = tracked_small is None or self._frames_since_detect >= self.redetect_every_frames
        if need_detect:
            fresh_small = _sensitivity_from_small_gray(gray, self.face_cascade_path, self.detail_weight, self.face_pad)
            self._map_small = fresh_small if tracked_small is None else np.maximum(fresh_small, tracked_small)
            self._frames_since_detect = 0
        else:
            self._map_small = tracked_small
            self._frames_since_detect += 1

        self._gray_small = gray
        full = cv2.resize(self._map_small, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(full, 0, 1).astype(np.float32)


def build_ink_map(ink, sensitivity_map, adaptive_strength, mode):
    """Turn a scalar --ink into a spatially-varying (height, width, 1) array
    when adaptive placement is on, so multiply/overlay blending reads
    darker/stronger exactly over `sensitivity_map`'s detected regions.

    The shift always moves `ink` further from that blend mode's own no-op
    point (1.0 for multiply, 0.5 for overlay) without ever crossing it,
    however large `adaptive_strength` is - crossing it would flip which
    side of the no-op point you're on and invert the effect.

    Returns `ink` unchanged for mode="alpha" or when adaptive placement
    isn't in use.
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
