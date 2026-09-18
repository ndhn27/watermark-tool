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

Requires: opencv-python (decoding/reading frames), numpy, and (for
audio-preserving, H.264 output) an ffmpeg binary on PATH.
"""

import os
import shutil
import subprocess
from collections import deque
from concurrent.futures import ProcessPoolExecutor

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


def _init_worker(mark_rgb01_bytes, mark_alpha01_bytes, shape, mode, ink):
    """Runs once per worker process at pool startup (not once per frame) -
    unpacks the shared watermark layer into that process's own globals so
    every frame task just reuses it instead of re-sending it each time.
    """
    global _worker_mark_rgb01, _worker_mark_alpha01, _worker_mode, _worker_ink
    h, w = shape
    _worker_mark_rgb01 = np.frombuffer(mark_rgb01_bytes, dtype=np.float32).reshape(h, w, 3)
    _worker_mark_alpha01 = np.frombuffer(mark_alpha01_bytes, dtype=np.float32).reshape(h, w, 1)
    _worker_mode = mode
    _worker_ink = ink


def _process_frame(frame_bgr_bytes, shape):
    """Watermark one frame. frame_bgr_bytes is the raw bytes of an HxWx3
    uint8 array in OpenCV's native BGR order; returns the same shape.
    """
    h, w, c = shape
    frame = np.frombuffer(frame_bgr_bytes, dtype=np.uint8).reshape(h, w, c)
    base01 = frame[..., ::-1].astype(np.float32) / 255.0  # BGR -> RGB, 0..1
    out01 = wmcore.blend_arrays(
        base01, _worker_mark_alpha01, mode=_worker_mode, ink=_worker_ink, mark_rgb01=_worker_mark_rgb01
    )
    out_bgr = np.clip(out01 * 255.0 + 0.5, 0, 255).astype(np.uint8)[..., ::-1]
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
    pool = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(mark_rgb01.tobytes(), mark_alpha01.tobytes(), (height, width), blend, ink),
    )

    max_in_flight = max(workers * 4, 4)
    pending = deque()
    progress = tqdm(total=frame_count, desc=os.path.basename(input_path), unit="frame", disable=not show_progress)

    def _drain_one():
        frame_bytes = pending.popleft().result()
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
            pending.append(pool.submit(_process_frame, frame.tobytes(), frame.shape))
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

    print(f"Saved: {output_path}")
