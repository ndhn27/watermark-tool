#!/usr/bin/env python3
"""
video.py - Apply the same diagonal-grid watermark used for stills to video
files, frame by frame, with the frame math parallelized across processes.

Pipeline:
  1. Build the watermark grid layer at the video's frame size (the same
     create_watermark_layer() used for images - see wmcore.py) - but NOT
     just once for the whole file. The video is split into fixed-length
     time segments (`jitter_refresh_seconds`, default a few seconds), and
     each segment gets its own freshly-generated layer with its own jitter
     seed (see `derive_seed` in wmcore.py). A single static layer reused
     for the entire clip would itself be a collusion/averaging-attack
     target - the base video content moves frame to frame but the mark
     wouldn't, so simply taking the per-pixel median (or mean) across many
     frames estimates that static pattern and lets it be subtracted, with
     no need for multiple different files the way the README's threat
     model otherwise assumes. Refreshing the pattern periodically means
     that attack only ever gets to average over one segment's worth of
     frames before the target changes underneath it. Pass
     `jitter_refresh_seconds=0` to fall back to the old single-static-layer
     behavior (e.g. for exact reproducibility with older output, or if
     the per-segment worker-pool rebuild below is unwelcome overhead on a
     short clip).
  2. Read frames sequentially from the source with OpenCV. Decoding stays
     single-threaded (cv2.VideoCapture isn't shareable across processes);
     that's fine, it's cheap next to the per-frame blend math.
  3. Hand each frame to a pool of worker processes that composite the
     watermark using the same blend_arrays() math as the image path, in a
     bounded pipeline (frames are submitted ahead of the writer so workers
     stay busy, but never more than a few batches ahead, to cap memory use).
     The pool is rebuilt at each segment boundary (draining all frames from
     the outgoing segment first) so workers pick up the new segment's mark
     arrays - simpler and safer than mutating shared memory a worker might
     be mid-read on, at the cost of a small pause every `jitter_refresh_seconds`.
  4. Stream the finished frames, in order, into an ffmpeg subprocess that
     re-encodes them to H.264 and muxes back the original audio track in
     one pass.

If ffmpeg isn't on PATH, this falls back to OpenCV's own writer: it still
works, but the output has no audio and uses a much less efficient codec.
Install ffmpeg for real use.

Content-adaptive placement (--adaptive) used to detect faces/high-detail
regions once, from frame 0, and reuse that single snapshot for the entire
clip - fine for a static shot, wrong the moment a subject or the camera
moves, since the boosted region just stays wherever it was first seen while
the actual content drifts out from under it. Two additions fix this without
touching the segment/worker-pool architecture above (mark arrays are still
only rebuilt at segment boundaries - see step 1 - so none of this adds
per-frame IPC cost):

  - `adaptive_track`: wmcore.AdaptiveTracker (see wmcore/adaptive.py) runs
    once per decoded frame, in the main process, entirely separately from
    the worker pool. Most frames it just carries the last detected map
    forward with cheap dense optical flow on a small downscaled image;
    periodically (`adaptive_redetect_seconds`) it re-runs the real face/
    detail detection to correct drift and catch new subjects. Whatever the
    tracker currently holds at the moment a segment boundary is reached is
    what that segment's mark/ink arrays get built from - so the sensitivity
    map a segment uses is always close to what's actually on screen for
    that segment, not a fossil of frame 0.
  - `scene_cut_detect`: a cheap frame-to-frame grayscale-difference check
    (also main-process, also every frame) flags hard cuts. A cut both tells
    the tracker to skip flow-warping (motion "across" a cut is meaningless)
    and forces an immediate segment boundary - new jitter pattern *and* a
    fresh detection pass - right at the cut, instead of dragging the old
    shot's mark and sensitivity map a few more seconds into the new shot
    until the next timed refresh happens to land.

Frame transport (step 3) uses a pair of shared-memory ring buffers rather
than pickling each frame through the worker pool's IPC pipe:
  - The main process writes each decoded frame directly into a slot of a
    shared "in" buffer, submits only that slot's (small) index to a worker,
    and the worker writes its result into the matching slot of a shared
    "out" buffer and returns just the index.
  - Without this, every frame crosses the pipe twice at full size (once as
    the call argument, once as the return value) via pickle - for a 4K
    frame that's ~50 MB of copying per frame just for IPC, which becomes
    the bottleneck on long/high-resolution video well before the actual
    blend math does. Only a handful of slots (`max_in_flight`, sized off
    `--workers`) are ever alive at once, cycled round-robin in the same
    order frames are submitted and drained, so a slot is never reused
    while a still-pending frame owns it.
  - If shared memory can't be created (e.g. some sandboxed/CI containers
    without a usable /dev/shm), this transparently falls back to the
    old pickle-per-frame path instead of failing outright.

Requires: opencv-python (decoding/reading frames), numpy, and (for
audio-preserving, H.264 output) an ffmpeg binary on PATH.

Invisible mark (--invisible --invisible-mode spread, i.e. `ss_key` given below):
embeds the SAME wmcore.embed_spread_spectrum() used for images into every
frame, inside the worker pool above (step 3) right after the visible grid is
composited - so it costs one extra DCT/inverse-DCT round trip per frame, not
a separate pass. The KEY rotates every `jitter_refresh_seconds` (the grid's
own cadence - see `ss_frame_index` below), the same anti-averaging reasoning
as the grid's jitter, but on its OWN fixed-interval schedule: it does NOT
follow the grid's scene-cut-triggered early boundaries, specifically so a
decoder can recompute which key a given frame used from nothing but that
frame's position + fps + `jitter_refresh_seconds` (all public, unlike
`ss_key` itself) without having to redetect scene cuts from the frames
itself. See wmapp.extraction / scan_invisible() below for the read-back
side, and bench/results/P4-NOTES.md for what was measured before this was
added (default alpha, per-frame survival under real ffmpeg transcodes,
why decode scans every frame instead of stopping at the first hit).
"""

import atexit
import os
import shutil
import subprocess
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field as _dc_field
from multiprocessing import shared_memory

import cv2
import numpy as np
from PIL import Image

import c2pa_provenance
import wmcore

try:
    from tqdm import tqdm
except ImportError:
    # requirements.txt lists tqdm, so this is a safety net rather than the
    # expected path - the tool still runs without it, just silently.
    class _NullProgress:
        def __init__(self, total=None, **kwargs):
            self.n = 0

        def update(self, n=1):
            self.n += n

        def close(self):
            pass

    def tqdm(iterable=None, total=None, **kwargs):
        if iterable is not None:
            return iterable
        return _NullProgress(total)


def _print_note(message, level="info", code=None):
    print(message)


_short_refresh_warned = False  # print the short-refresh-interval note at most once per process


# ---------------------------------------------------------------------------
# Per-frame worker (runs in a separate process - see wmcore.py's module
# docstring for why this can't be a closure or live in the __main__ script)
# ---------------------------------------------------------------------------

_worker_mark_rgb01 = None
_worker_mark_alpha01 = None
_worker_mode = None
_worker_ink = None
_worker_ss_payload = None    # text to embed with the invisible mark, or None (no invisible mark this run)
_worker_ss_alpha = None

# Only set up when the shared-memory path is used (_init_worker_shm below).
_worker_in_shm = None
_worker_out_shm = None
_worker_in_arr = None
_worker_out_arr = None


def _init_worker_common(mark_rgb01_bytes, mark_alpha01_bytes, shape, mode, ink, ss_payload, ss_alpha):
    global _worker_mark_rgb01, _worker_mark_alpha01, _worker_mode, _worker_ink, _worker_ss_payload, _worker_ss_alpha
    h, w = shape
    _worker_mark_rgb01 = np.frombuffer(mark_rgb01_bytes, dtype=np.float32).reshape(h, w, 3)
    _worker_mark_alpha01 = np.frombuffer(mark_alpha01_bytes, dtype=np.float32).reshape(h, w, 1)
    _worker_mode = mode
    _worker_ink = ink
    _worker_ss_payload = ss_payload
    _worker_ss_alpha = ss_alpha


def _blend_frame(frame_rgb_view, ss_key):
    """Shared math: frame_rgb_view is an HxWx3 uint8 array already in BGR
    order (OpenCV's native order); returns an HxWx3 uint8 array, also BGR.
    `ss_key` is this FRAME's already-derived per-segment invisible-mark key
    (None if no invisible mark is being embedded this run - see
    _worker_ss_payload) - computed in the main process (see `ss_frame_index`
    in apply_watermark_video) and passed in per submit() call, since it
    changes more often than the pool is rebuilt (see the module docstring).
    """
    base01 = frame_rgb_view[..., ::-1].astype(np.float32) / 255.0  # BGR -> RGB, 0..1
    out01 = wmcore.blend_arrays(
        base01, _worker_mark_alpha01, mode=_worker_mode, ink=_worker_ink, mark_rgb01=_worker_mark_rgb01
    )
    rgb_u8 = np.clip(out01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if _worker_ss_payload is not None and ss_key is not None:
        marked = wmcore.embed_spread_spectrum(
            Image.fromarray(rgb_u8, "RGB").convert("RGBA"), _worker_ss_payload, ss_key, alpha=_worker_ss_alpha,
        )
        rgb_u8 = np.asarray(marked.convert("RGB"), dtype=np.uint8)
    return rgb_u8[..., ::-1]


# --- Shared-memory path ------------------------------------------------

def _init_worker_shm(mark_rgb01_bytes, mark_alpha01_bytes, shape, mode, ink, in_shm_name, out_shm_name, slot_count,
                      ss_payload, ss_alpha):
    """Pool initializer for the shared-memory transport. Runs once per
    worker process, not once per frame.
    """
    global _worker_in_shm, _worker_out_shm, _worker_in_arr, _worker_out_arr
    _init_worker_common(mark_rgb01_bytes, mark_alpha01_bytes, shape, mode, ink, ss_payload, ss_alpha)
    h, w = shape
    _worker_in_shm = shared_memory.SharedMemory(name=in_shm_name)
    _worker_out_shm = shared_memory.SharedMemory(name=out_shm_name)
    _worker_in_arr = np.ndarray((slot_count, h, w, 3), dtype=np.uint8, buffer=_worker_in_shm.buf)
    _worker_out_arr = np.ndarray((slot_count, h, w, 3), dtype=np.uint8, buffer=_worker_out_shm.buf)
    # Unregister (without unlinking) so multiprocessing's resource_tracker
    # in this worker process doesn't warn about "leaked" shared memory when
    # the worker exits - the main process owns the unlink.
    atexit.register(_worker_in_shm.close)
    atexit.register(_worker_out_shm.close)


def _process_frame_shm(slot, ss_key):
    """Watermark the frame sitting in input slot `slot`, writing the result
    into the same slot of the output buffer. Only the slot index and this
    frame's invisible-mark key (a small int, or None) cross the pool's IPC
    pipe in either direction.
    """
    _worker_out_arr[slot] = _blend_frame(_worker_in_arr[slot], ss_key)
    return slot


# --- Pickle-per-frame fallback (used only if shared memory setup fails) -

def _init_worker_pickle(mark_rgb01_bytes, mark_alpha01_bytes, shape, mode, ink, ss_payload, ss_alpha):
    _init_worker_common(mark_rgb01_bytes, mark_alpha01_bytes, shape, mode, ink, ss_payload, ss_alpha)


def _process_frame_pickle(frame_bgr_bytes, shape, ss_key):
    h, w, c = shape
    frame = np.frombuffer(frame_bgr_bytes, dtype=np.uint8).reshape(h, w, c)
    out_bgr = _blend_frame(frame, ss_key)
    return np.ascontiguousarray(out_bgr).tobytes()


# ---------------------------------------------------------------------------
# Output encoding: ffmpeg (video + original audio, one pass) or OpenCV fallback
# ---------------------------------------------------------------------------

def _ffmpeg_encode_pipe(ffmpeg_bin, output_path, width, height, fps, audio_source):
    cmd = [
        ffmpeg_bin, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",                      # watermarked frames, piped in as raw video
        "-i", audio_source,             # original file, for its audio track only
        "-map", "0:v:0", "-map", "1:a:0?",   # "?" = skip the audio map if source has none
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _build_mark_arrays(
    size, text, font_size, opacity, angle, spacing, font_path, seed,
    sensitivity_map=None, adaptive_strength=0.6, infill_spacing_divisor=2.0,
):
    mark_layer = wmcore.create_watermark_layer(
        size, text, font_size, opacity, angle, spacing, font_path, seed=seed,
        sensitivity_map=sensitivity_map, adaptive_strength=adaptive_strength,
        infill_spacing_divisor=infill_spacing_divisor,
    )
    return wmcore.watermark_layer_to_arrays(mark_layer)


def apply_watermark_video(
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
    workers=None,
    show_progress=True,
    use_shared_memory=True,
    jitter_refresh_seconds=4.0,
    adaptive=False,
    adaptive_strength=0.6,
    adaptive_detail_weight=0.5,
    face_cascade=None,
    adaptive_infill_divisor=2.0,
    adaptive_track=True,
    adaptive_redetect_seconds=1.0,
    scene_cut_detect=True,
    scene_cut_threshold=0.12,
    c2pa=False,
    c2pa_cert=None,
    c2pa_key=None,
    c2pa_alg=c2pa_provenance.DEFAULT_ALG,
    c2pa_title=None,
    author=None,
    wm_id=None,
    ss_key=None,
    ss_alpha=None,
    ss_payload=None,
    notify=None,
):
    """Watermark a video, frame by frame, in parallel. Every user-facing note
    ("ffmpeg not found...", the short-refresh warning...) goes through
    `notify(message, level="info", code=None)` - by default a plain print, so
    the CLI is unchanged; wmapp passes a collector so the same notes become
    part of the result an API or web client receives. Printing "Saved: ..." is
    the caller's job (it knows the output path already).

    `ss_key`/`ss_payload` add the invisible spread-spectrum mark (see the
    module docstring's "Invisible mark" section) - both must be given
    together (ss_payload is the text; typically a watermark ID) or both left
    None (no invisible mark, the old behavior). `ss_alpha` defaults to
    wmcore.DEFAULT_SS_ALPHA_VIDEO if not given."""
    notify = notify or _print_note
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    workers = workers or os.cpu_count() or 1
    ss_alpha = wmcore.DEFAULT_SS_ALPHA_VIDEO if (ss_payload is not None and ss_alpha is None) else ss_alpha

    # frames_per_segment=None means "one segment for the whole video" - the
    # old, static-mark behavior - when jitter_refresh_seconds is disabled.
    if jitter_refresh_seconds and jitter_refresh_seconds > 0:
        frames_per_segment = max(1, round(fps * jitter_refresh_seconds))
    else:
        frames_per_segment = None

    global _short_refresh_warned
    if frames_per_segment is not None and jitter_refresh_seconds < 1.0 and not _short_refresh_warned:
        notify(
            f"Note: --jitter-refresh-seconds {jitter_refresh_seconds:g} rebuilds the worker "
            f"pool very often (every ~{frames_per_segment} frames here). Each rebuild has a "
            f"real, roughly fixed cost that gets worse with more --workers (currently {workers}) "
            "and matters more on short clips, where rebuild overhead is a bigger fraction of "
            "total runtime. Benchmark on your own machine before committing to sub-1s refresh "
            "for a long/production run - and only go this short if you actually need to defend "
            "against an adversary who can median/average frames over a window that narrow.",
            code="jitter_refresh_short",
        )
        _short_refresh_warned = True

    # Adaptive placement: an AdaptiveTracker (see wmcore/adaptive.py) tracks
    # faces/high-detail regions across the whole clip - cheap per-frame
    # optical-flow tracking between periodic full re-detections - instead of
    # detecting once from frame 0 and reusing that snapshot forever. The
    # first frame itself is NOT discarded - _first_frame_pending below feeds
    # it back into the normal per-frame loop as frame 0.
    first_ok, first_frame = cap.read()
    _first_frame_pending = True
    sensitivity_map = None
    tracker = None
    if adaptive:
        if first_ok:
            if adaptive_track:
                redetect_every_frames = max(1, round(fps * adaptive_redetect_seconds))
                tracker = wmcore.AdaptiveTracker(
                    face_cascade_path=face_cascade, detail_weight=adaptive_detail_weight,
                    redetect_every_frames=redetect_every_frames, cut_threshold=scene_cut_threshold,
                )
                sensitivity_map = tracker.update(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
            else:
                sensitivity_map = wmcore.compute_sensitivity_map(
                    cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB),
                    face_cascade_path=face_cascade, detail_weight=adaptive_detail_weight,
                )
        else:
            notify(f"Note: {input_path} has no readable frames; skipping --adaptive detection.",
                   level="warning", code="adaptive_no_frames")

    # Scene-cut detection runs independently of --adaptive: even a plain
    # jittered grid benefits from its mark refresh landing exactly on a real
    # content boundary rather than drifting a few frames into the new shot
    # before the next timed refresh happens to fire. When --adaptive-track is
    # also on, the tracker (above) does its own, separate cut check on the
    # same frames - see wmcore.AdaptiveTracker.last_cut - so this one is
    # skipped then to avoid computing the same grayscale diff twice.
    _scene_cut_gray = None
    if scene_cut_detect and not (adaptive and adaptive_track) and first_ok:
        small = cv2.resize(first_frame, (max(1, width // 4), max(1, height // 4)), interpolation=cv2.INTER_AREA)
        _scene_cut_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def _is_scene_cut(frame_bgr):
        """Standalone cut check for when the adaptive tracker isn't already
        doing this same comparison itself (see the note above)."""
        nonlocal _scene_cut_gray
        small = cv2.resize(frame_bgr, (max(1, width // 4), max(1, height // 4)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        cut = False
        if _scene_cut_gray is not None and _scene_cut_gray.shape == gray.shape:
            cut = float(np.mean(cv2.absdiff(gray, _scene_cut_gray))) / 255.0 > scene_cut_threshold
        _scene_cut_gray = gray
        return cut

    def _build_segment_arrays(seg_seed, seg_sensitivity_map):
        mrgb, malpha = _build_mark_arrays(
            (width, height), text, font_size, opacity, angle, spacing, font_path,
            seed=seg_seed, sensitivity_map=seg_sensitivity_map, adaptive_strength=adaptive_strength,
            infill_spacing_divisor=adaptive_infill_divisor,
        )
        seg_ink = wmcore.build_ink_map(ink, seg_sensitivity_map, adaptive_strength, blend)
        return mrgb, malpha, seg_ink

    segment_index = 0
    mark_rgb01, mark_alpha01, ink_for_pool = _build_segment_arrays(
        wmcore.derive_seed(seed, 0) if frames_per_segment else seed, sensitivity_map,
    )

    ffmpeg_bin = shutil.which("ffmpeg")
    writer = None
    proc = None
    if ffmpeg_bin:
        proc = _ffmpeg_encode_pipe(ffmpeg_bin, output_path, width, height, fps, input_path)
    else:
        notify(
            "Note: ffmpeg not found on PATH - falling back to OpenCV's writer. "
            "Output will have NO audio and use a lower-quality codec. "
            "Install ffmpeg (https://ffmpeg.org) for audio-preserving, H.264 output.",
            level="warning", code="ffmpeg_missing",
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    max_in_flight = max(workers * 4, 4)

    in_shm = out_shm = in_arr = out_arr = None
    if use_shared_memory:
        try:
            slot_nbytes = height * width * 3
            in_shm = shared_memory.SharedMemory(create=True, size=max_in_flight * slot_nbytes)
            out_shm = shared_memory.SharedMemory(create=True, size=max_in_flight * slot_nbytes)
            in_arr = np.ndarray((max_in_flight, height, width, 3), dtype=np.uint8, buffer=in_shm.buf)
            out_arr = np.ndarray((max_in_flight, height, width, 3), dtype=np.uint8, buffer=out_shm.buf)
        except OSError as exc:
            notify(f"Note: couldn't set up shared memory for video frames ({exc}); falling back to a slower per-frame transport.",
                   code="shared_memory_fallback")
            if in_shm is not None:
                in_shm.close()
                in_shm.unlink()
            in_shm = out_shm = in_arr = out_arr = None

    use_shm = in_arr is not None

    def _make_pool(mrgb, malpha, pool_ink):
        # `pool_ink` is this segment's own build_ink_map() result (a scalar
        # or a per-pixel array - see _build_segment_arrays), NOT the raw
        # --ink value: with --adaptive-track, that result changes every
        # segment as the tracked sensitivity map moves, so it has to be
        # passed in fresh each time rather than closed over once.
        if use_shm:
            return ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker_shm,
                initargs=(
                    mrgb.tobytes(), malpha.tobytes(), (height, width), blend, pool_ink,
                    in_shm.name, out_shm.name, max_in_flight, ss_payload, ss_alpha,
                ),
            )
        return ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker_pickle,
            initargs=(mrgb.tobytes(), malpha.tobytes(), (height, width), blend, pool_ink, ss_payload, ss_alpha),
        )

    pool = _make_pool(mark_rgb01, mark_alpha01, ink_for_pool)

    pending = deque()  # (slot_or_None, future)
    next_slot = 0
    frames_in_segment = 0
    ss_frame_index = 0  # separate, non-resetting counter for the invisible mark's OWN fixed-interval
                         # key schedule - deliberately not tied to frames_in_segment/segment_index above,
                         # which also rotate early on a scene cut (see the module docstring: the
                         # invisible mark's schedule must stay computable from position + fps alone).
    progress = tqdm(total=frame_count, desc=os.path.basename(input_path), unit="frame", disable=not show_progress)

    def _drain_one():
        slot, fut = pending.popleft()
        if use_shm:
            fut.result()  # raises on worker error; return value is just `slot` again
            frame_view = out_arr[slot]
            if writer is not None:
                writer.write(frame_view)
            else:
                proc.stdin.write(frame_view.tobytes())
        else:
            frame_bytes = fut.result()
            if writer is not None:
                arr = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width, 3)
                writer.write(arr)
            else:
                proc.stdin.write(frame_bytes)
        progress.update(1)

    try:
        while True:
            if _first_frame_pending:
                ok, frame = first_ok, first_frame
                _first_frame_pending = False
                is_first_frame = True
            else:
                ok, frame = cap.read()
                is_first_frame = False
            if not ok:
                break

            # Advance the adaptive tracker / scene-cut check with THIS frame
            # before deciding on a segment boundary below, so a triggered
            # rebuild uses a sensitivity map current as of the frame it's
            # about to apply to - not the previous segment's. Skipped
            # entirely when segments are off (frames_per_segment is None):
            # with only one segment for the whole clip, nothing would ever
            # read an updated map again, so tracking it would just be wasted
            # per-frame CPU in the main decode process (see the module
            # docstring's --jitter-refresh-seconds 0 note).
            cut_now = False
            if not is_first_frame and frames_per_segment is not None:
                if adaptive and adaptive_track and tracker is not None:
                    sensitivity_map = tracker.update(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    cut_now = scene_cut_detect and tracker.last_cut
                elif scene_cut_detect:
                    cut_now = _is_scene_cut(frame)

            boundary_due = frames_per_segment is not None and (
                frames_in_segment >= frames_per_segment or (cut_now and frames_in_segment > 0)
            )
            if boundary_due:
                # Segment boundary: finish every frame still in flight under
                # the OLD mark before workers ever see the new one, then
                # swap pools. This keeps the two segments' patterns from
                # ever blending mid-frame - simpler to reason about than
                # having a worker mid-read on mark data another process is
                # concurrently rewriting. A scene cut (cut_now) forces this
                # early, ahead of the fixed jitter_refresh_seconds cadence -
                # see the module docstring.
                while pending:
                    _drain_one()
                pool.shutdown(wait=True)
                segment_index += 1
                mark_rgb01, mark_alpha01, ink_for_pool = _build_segment_arrays(
                    wmcore.derive_seed(seed, segment_index), sensitivity_map,
                )
                pool = _make_pool(mark_rgb01, mark_alpha01, ink_for_pool)
                frames_in_segment = 0

            if use_shm:
                slot = next_slot
                next_slot = (next_slot + 1) % max_in_flight
                in_arr[slot] = frame
                ss_key_for_frame = (
                    wmcore.derive_seed(ss_key, ss_frame_index // frames_per_segment)
                    if (ss_key is not None and frames_per_segment is not None) else ss_key
                )
                pending.append((slot, pool.submit(_process_frame_shm, slot, ss_key_for_frame)))
            else:
                ss_key_for_frame = (
                    wmcore.derive_seed(ss_key, ss_frame_index // frames_per_segment)
                    if (ss_key is not None and frames_per_segment is not None) else ss_key
                )
                pending.append((None, pool.submit(_process_frame_pickle, frame.tobytes(), frame.shape, ss_key_for_frame)))
            frames_in_segment += 1
            ss_frame_index += 1
            if len(pending) >= max_in_flight:
                _drain_one()
        while pending:
            _drain_one()
    finally:
        progress.close()
        cap.release()
        pool.shutdown(wait=True)
        if writer is not None:
            writer.release()
        if proc is not None:
            proc.stdin.close()
            stderr = proc.stderr.read() if proc.stderr else b""
            returncode = proc.wait()
            if returncode != 0:
                raise RuntimeError(
                    f"ffmpeg exited with code {returncode} while encoding {output_path}:\n"
                    f"{stderr.decode(errors='replace')[:2000]}"
                )
        if in_shm is not None:
            in_shm.close()
            in_shm.unlink()
        if out_shm is not None:
            out_shm.close()
            out_shm.unlink()

    if c2pa:
        # Signs the fully-muxed output on disk - has to happen after the
        # ffmpeg/writer `finally` block above closes, not before, since
        # that's the point the actual final video bytes exist. See
        # c2pa_provenance.sign_file_or_note() for why a signing failure is a
        # note rather than an exception: a video that took real CPU-minutes to
        # watermark shouldn't be discarded over a C2PA problem alone.
        c2pa_provenance.sign_file_or_note(
            output_path, title=c2pa_title, author=author, parent_path=input_path,
            cert_path=c2pa_cert, key_path=c2pa_key, alg=c2pa_alg, wm_id=wm_id,
            notify=lambda message: notify(message, code="c2pa_failed", level="warning"),
        )


# ---------------------------------------------------------------------------
# Invisible mark: read-back
# ---------------------------------------------------------------------------

@dataclass
class VideoScanResult:
    """The full breakdown of one scan_invisible() run against one candidate
    `ss_key` - deliberately not collapsed to a single yes/no (see
    scan_invisible's docstring and bench/results/P4-NOTES.md)."""
    frames_total: int
    frames_checked: int
    hits: int                          # frames that decoded ANY watermark ID
    ids_seen: dict = _dc_field(default_factory=dict)   # wm_id -> how many frames decoded it
    fps: float = 0.0
    ss_key: int = None
    jitter_refresh_seconds: float = 4.0
    regrid_hits: int = 0               # of `hits`: frames that only decoded after the crop re-alignment (wmcore/resync.py)

    @property
    def wm_id(self):
        """The most-seen ID, or None if nothing decoded. Normally there is
        only one distinct ID in ids_seen at all (the payload never changes
        mid-video, only the key that reads it back does) - more than one
        means a handful of frames decoded a DIFFERENT id than the rest,
        which for a real mark should not happen and is worth a second look
        (a wrong --ss-key can still occasionally decode junk that happens to
        pass the ID checksum; see README on --ss-key not being a secrecy
        mechanism)."""
        if not self.ids_seen:
            return None
        return max(self.ids_seen, key=self.ids_seen.get)

    def to_dict(self):
        return {"frames_total": self.frames_total, "frames_checked": self.frames_checked,
                "hits": self.hits, "ids_seen": dict(self.ids_seen), "fps": self.fps,
                "ss_key": self.ss_key, "jitter_refresh_seconds": self.jitter_refresh_seconds,
                "wm_id": self.wm_id, "regrid_hits": self.regrid_hits}


def scan_invisible(path, ss_key, jitter_refresh_seconds=4.0, max_length=None, resync=True):
    """Scan EVERY frame of `path` for a wmcore.embed_spread_spectrum() watermark
    ID, recomputing for each frame the same per-segment key
    apply_watermark_video() would have derived for a frame at that position
    (see the module docstring's "Invisible mark" section) - NOT just trying
    `ss_key` unchanged against every frame.

    Deliberately does not stop at the first frame that decodes: a still image
    gets exactly one decode attempt, a video gets one PER FRAME, so the
    number that matters is "did any frame decode", not "what fraction of
    frames decoded" - and reporting the full breakdown (frames_checked, hits,
    ids_seen) rather than collapsing it to that one yes/no is the same
    "don't hide the number" choice wmapp.extraction already makes for
    images' per-layer report. See bench/results/P4-NOTES.md for the measured
    per-frame survival numbers this is built on.

    Does NOT replicate the grid's scene-cut-triggered early key rotation
    (apply_watermark_video's `cut_now`/`boundary_due`) - only its fixed
    `jitter_refresh_seconds` interval (see `ss_frame_index` there). Frames
    within a few positions of a real scene cut may therefore use a key one
    segment off from what was actually embedded and simply not decode; nothing
    special is done about that here (c.f. P4-NOTES.md §4/§7 - not yet solved,
    not silently patched over). `jitter_refresh_seconds` must match what was
    used at embed time (its own default, 4.0, unless that was overridden) -
    unlike `ss_key`, it is not a secret, but the segment math still has to
    agree to land on the same derived key.

    `resync` (default on): a frame the plain decode could not read is tried again as a CROPPED frame
    (wmcore/resync.py: the 8x8 block grid re-found, the row length re-derived), unmirrored only - a video is
    cropped far more often than mirrored, and the search runs on every frame the plain decode misses.
    VideoScanResult.regrid_hits counts the frames read that way, so a video that only reads after
    re-alignment is visible as such rather than folded into `hits`.

    Returns a VideoScanResult. Raises RuntimeError if `path` can't be opened."""
    from wmcore.invisible import MAX_PLAUSIBLE_LENGTH
    max_length = MAX_PLAUSIBLE_LENGTH if max_length is None else max_length
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames_per_segment = (
            max(1, round(fps * jitter_refresh_seconds)) if jitter_refresh_seconds and jitter_refresh_seconds > 0
            else None
        )
        ids_seen = {}
        hits = checked = regrid_hits = i = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            key_i = (wmcore.derive_seed(ss_key, i // frames_per_segment)
                     if frames_per_segment is not None else ss_key)
            img = Image.fromarray(frame_bgr[..., ::-1], "RGB")
            corrs = wmcore.spread_spectrum_block_correlations(img, key_i)
            found_id = wmcore.find_spread_spectrum_id(corrs)
            if found_id is None and resync:
                match = wmcore.find_spread_spectrum_id_resync(img, key_i, orientations=("none",), max_seconds=5.0)
                if match.text is not None:
                    found_id = match.text
                    regrid_hits += 1
            checked += 1
            if found_id is not None:
                hits += 1
                ids_seen[found_id] = ids_seen.get(found_id, 0) + 1
            i += 1
    finally:
        cap.release()
    return VideoScanResult(frames_total=i, frames_checked=checked, hits=hits, ids_seen=ids_seen,
                           fps=fps, ss_key=ss_key, jitter_refresh_seconds=jitter_refresh_seconds,
                           regrid_hits=regrid_hits)

