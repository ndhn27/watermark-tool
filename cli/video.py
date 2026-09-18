#!/usr/bin/env python3
"""
video.py - Apply the same diagonal-grid watermark used for stills to video
files, frame by frame, with the frame math parallelized across processes.

Pipeline:
  1. Build the watermark grid layer once, at the video's frame size (the
     same create_watermark_layer() used for images - see wmcore.py). One
     build, reused for every frame - a video is just a still grid stamped
     onto a moving base.
  2. Read frames sequentially from the source with OpenCV. Decoding stays
     single-threaded (cv2.VideoCapture isn't shareable across processes);
     that's fine, it's cheap next to the per-frame blend math.
  3. Hand each frame to a pool of worker processes that composite the
     watermark using the same blend_arrays() math as the image path, in a
     bounded pipeline (frames are submitted ahead of the writer so workers
     stay busy, but never more than a few batches ahead, to cap memory use).
  4. Stream the finished frames, in order, into an ffmpeg subprocess that
     re-encodes them to H.264 and muxes back the original audio track in
     one pass.

If ffmpeg isn't on PATH, this falls back to OpenCV's own writer: it still
works, but the output has no audio and uses a much less efficient codec.
Install ffmpeg for real use.

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
"""

import atexit
import os
import shutil
import subprocess
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

import cv2
import numpy as np

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


# ---------------------------------------------------------------------------
# Per-frame worker (runs in a separate process - see wmcore.py's module
# docstring for why this can't be a closure or live in the __main__ script)
# ---------------------------------------------------------------------------

_worker_mark_rgb01 = None
_worker_mark_alpha01 = None
_worker_mode = None
_worker_ink = None

# Only set up when the shared-memory path is used (_init_worker_shm below).
_worker_in_shm = None
_worker_out_shm = None
_worker_in_arr = None
_worker_out_arr = None


def _init_worker_common(mark_rgb01_bytes, mark_alpha01_bytes, shape, mode, ink):
    global _worker_mark_rgb01, _worker_mark_alpha01, _worker_mode, _worker_ink
    h, w = shape
    _worker_mark_rgb01 = np.frombuffer(mark_rgb01_bytes, dtype=np.float32).reshape(h, w, 3)
    _worker_mark_alpha01 = np.frombuffer(mark_alpha01_bytes, dtype=np.float32).reshape(h, w, 1)
    _worker_mode = mode
    _worker_ink = ink


def _blend_frame(frame_rgb_view):
    """Shared math: frame_rgb_view is an HxWx3 uint8 array already in BGR
    order (OpenCV's native order); returns an HxWx3 uint8 array, also BGR.
    """
    base01 = frame_rgb_view[..., ::-1].astype(np.float32) / 255.0  # BGR -> RGB, 0..1
    out01 = wmcore.blend_arrays(
        base01, _worker_mark_alpha01, mode=_worker_mode, ink=_worker_ink, mark_rgb01=_worker_mark_rgb01
    )
    return np.clip(out01 * 255.0 + 0.5, 0, 255).astype(np.uint8)[..., ::-1]


# --- Shared-memory path ------------------------------------------------

def _init_worker_shm(mark_rgb01_bytes, mark_alpha01_bytes, shape, mode, ink, in_shm_name, out_shm_name, slot_count):
    """Pool initializer for the shared-memory transport. Runs once per
    worker process, not once per frame.
    """
    global _worker_in_shm, _worker_out_shm, _worker_in_arr, _worker_out_arr
    _init_worker_common(mark_rgb01_bytes, mark_alpha01_bytes, shape, mode, ink)
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


def _process_frame_shm(slot):
    """Watermark the frame sitting in input slot `slot`, writing the result
    into the same slot of the output buffer. Only the slot index (a small
    int) crosses the pool's IPC pipe in either direction.
    """
    _worker_out_arr[slot] = _blend_frame(_worker_in_arr[slot])
    return slot


# --- Pickle-per-frame fallback (used only if shared memory setup fails) -

def _init_worker_pickle(mark_rgb01_bytes, mark_alpha01_bytes, shape, mode, ink):
    _init_worker_common(mark_rgb01_bytes, mark_alpha01_bytes, shape, mode, ink)


def _process_frame_pickle(frame_bgr_bytes, shape):
    h, w, c = shape
    frame = np.frombuffer(frame_bgr_bytes, dtype=np.uint8).reshape(h, w, c)
    out_bgr = _blend_frame(frame)
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
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    mark_layer = wmcore.create_watermark_layer(
        (width, height), text, font_size, opacity, angle, spacing, font_path, seed=seed
    )
    mark_rgb01, mark_alpha01 = wmcore.watermark_layer_to_arrays(mark_layer)

    ffmpeg_bin = shutil.which("ffmpeg")
    writer = None
    proc = None
    if ffmpeg_bin:
        proc = _ffmpeg_encode_pipe(ffmpeg_bin, output_path, width, height, fps, input_path)
    else:
        print(
            "Note: ffmpeg not found on PATH - falling back to OpenCV's writer. "
            "Output will have NO audio and use a lower-quality codec. "
            "Install ffmpeg (https://ffmpeg.org) for audio-preserving, H.264 output."
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    workers = workers or os.cpu_count() or 1
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
            print(f"Note: couldn't set up shared memory for video frames ({exc}); falling back to a slower per-frame transport.")
            if in_shm is not None:
                in_shm.close()
                in_shm.unlink()
            in_shm = out_shm = in_arr = out_arr = None

    use_shm = in_arr is not None
    if use_shm:
        pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker_shm,
            initargs=(
                mark_rgb01.tobytes(), mark_alpha01.tobytes(), (height, width), blend, ink,
                in_shm.name, out_shm.name, max_in_flight,
            ),
        )
    else:
        pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker_pickle,
            initargs=(mark_rgb01.tobytes(), mark_alpha01.tobytes(), (height, width), blend, ink),
        )

    pending = deque()  # (slot_or_None, future)
    next_slot = 0
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
            ok, frame = cap.read()
            if not ok:
                break
            if use_shm:
                slot = next_slot
                next_slot = (next_slot + 1) % max_in_flight
                in_arr[slot] = frame
                pending.append((slot, pool.submit(_process_frame_shm, slot)))
            else:
                pending.append((None, pool.submit(_process_frame_pickle, frame.tobytes(), frame.shape)))
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

    print(f"Saved: {output_path}")
