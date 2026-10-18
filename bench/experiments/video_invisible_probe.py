#!/usr/bin/env python3
"""P4 (bước 3/5) - Đo TRƯỚC khi thêm gì vào cli/video.py: nếu ta nhúng dấu ẩn
block-DCT spread-spectrum (wmcore.embed_spread_spectrum(), cái ảnh --invisible
--invisible-mode spread đang dùng) vào TỪNG khung hình y hệt cách nhúng ảnh,
rồi mã hoá thật bằng ffmpeg (H.264 CRF cao, đổi fps, crop, letterbox), dấu đó
còn đọc được ở bao nhiêu % khung hình?

Đây CHƯA đụng vào cli/video.py - chỉ mô phỏng "điều gì sẽ xảy ra nếu" trước,
đúng theo yêu cầu của HANDOFF v23 mục 2, bước 3. Kết quả quyết định video.py
nên nhúng thế nào (mỗi khung độc lập? theo segment như lớp jitter hiện có?
alpha mặc định 12.0 có đủ không?).

Video tổng hợp: quét (pan) một khung nhìn qua một ảnh "pink" lớn hơn để có
chuyển động thật giữa các khung, không phải nhiễu độc lập mỗi khung (nhiễu độc
lập không giống video thật, nơi khung liền kề tương quan cao).

Cách dùng:
    python3 bench/experiments/video_invisible_probe.py
"""

import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in (".", "..", os.path.join("..", "..", "cli")):
    p = os.path.abspath(os.path.join(HERE, sub))
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import bench_corpus as C  # noqa: E402
import wmcore  # noqa: E402
from wmcore import wmid  # noqa: E402

FFMPEG = "ffmpeg"
SEED = 20260922 + 4_000_000
W, H = 640, 480
FPS = 24
DURATION_S = 2.5
N_FRAMES = round(FPS * DURATION_S)  # 60 frames


def _make_pan_source(seed, w, h, n_frames, pad=160):
    """A big pink-noise canvas, panned across to give correlated frame-to-frame
    motion (a static scene with camera pan, not independent noise per frame -
    see module docstring). Returns a list of (h, w, 3) uint8 RGB arrays."""
    big = np.asarray(C._pink(w + pad, h + pad, seed).convert("RGB"), dtype=np.uint8)
    frames = []
    max_off = pad
    for i in range(n_frames):
        # smooth back-and-forth pan so no two frames are byte-identical
        t = i / max(1, n_frames - 1)
        off = int(round(max_off * 0.5 * (1 - np.cos(2 * np.pi * t))))
        frames.append(np.ascontiguousarray(big[off:off + h, off:off + w]))
    return frames


def _run_ffmpeg(args, raw_frames_iter, w, h, fps):
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        *args,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in raw_frames_iter:
        proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    proc.stdin.close()
    stderr = proc.stderr.read()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed ({rc}): {stderr.decode(errors='replace')[:800]}")


def _read_frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        out.append(frame_bgr[..., ::-1])  # BGR -> RGB
    cap.release()
    return out


def _decode_rate(frames, key, sample_every=1):
    """Fraction of sampled frames from which the watermark ID decodes."""
    hits, n = 0, 0
    for i in range(0, len(frames), sample_every):
        img = Image.fromarray(frames[i], "RGB")
        corrs = wmcore.spread_spectrum_block_correlations(img, key)
        wm_id = wmcore.find_spread_spectrum_id(corrs)
        n += 1
        if wm_id is not None:
            hits += 1
    return hits, n


ATTACKS = {
    # name: (ffmpeg -vf filter or None, ffmpeg extra args)
    "crf18":        (None, ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]),
    "crf23":        (None, ["-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p"]),
    "crf32":        (None, ["-c:v", "libx264", "-preset", "medium", "-crf", "32", "-pix_fmt", "yuv420p"]),
    "crf40":        (None, ["-c:v", "libx264", "-preset", "medium", "-crf", "40", "-pix_fmt", "yuv420p"]),
    "fps_half":     (None, ["-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-r", str(FPS // 2)]),
    # crop_top: remove 40px (5 blocks) from the TOP only, width unchanged - block-column count
    # preserved, same shape the image benchmark's crop_top_64 covers (gate=True there).
    "crop_top_40":  (f"crop={W}:{H - 40}:0:40", ["-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p"]),
    # crop_sides: remove 40px from EACH side (left+right) - changes the block-COLUMN count,
    # the case the image code documents as out of scope (crop_topleft_64, gate=False).
    # Included here to measure, not assume, how badly that hurts on video.
    "crop_sides_40": (f"crop={W - 80}:{H}:40:0", ["-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p"]),
    # letterbox: pad 40px black bars top+bottom (simulates a repost that letterboxes to a
    # different aspect ratio) - multiple of 8, so it only shifts the block raster's phase,
    # which find_spread_spectrum_id's 112-phase search is designed to absorb (see module
    # docstring reasoning) - measuring whether that holds in practice, not just in theory.
    "letterbox_40": (f"pad={W}:{H + 80}:0:40:black", ["-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p"]),
}


def main():
    t0 = time.time()
    wm_id = wmid.generate()
    key = wmcore.derive_seed(SEED, "video_probe")
    print(f"wm_id={wm_id} key={key} frames={N_FRAMES} size={W}x{H} fps={FPS}")

    print("Building synthetic panned source + embedding per-frame (same embed_spread_spectrum() as images)...")
    src_frames = _make_pan_source(SEED, W, H, N_FRAMES)
    t_embed0 = time.time()
    wm_frames = []
    for f in src_frames:
        out = wmcore.embed_spread_spectrum(Image.fromarray(f, "RGB").convert("RGBA"), wm_id, key)
        wm_frames.append(np.asarray(out.convert("RGB"), dtype=np.uint8))
    embed_ms_per_frame = (time.time() - t_embed0) * 1000 / len(wm_frames)
    print(f"  embed: {embed_ms_per_frame:.1f} ms/frame (single-process, no --workers pooling)")

    # Sanity: does the mark decode on the RAW (never-transcoded) watermarked frames?
    hits, n = _decode_rate(wm_frames, key)
    print(f"[đã xác minh] raw watermarked frames, no encode at all: {hits}/{n}")

    with tempfile.TemporaryDirectory() as tmp:
        results = []
        for name, (vf, extra) in ATTACKS.items():
            out_path = os.path.join(tmp, f"{name}.mp4")
            args = (["-vf", vf] if vf else []) + extra + [out_path]
            t1 = time.time()
            _run_ffmpeg(args, wm_frames, W, H, FPS)
            enc_s = time.time() - t1
            frames_back = _read_frames(out_path)
            hits, n = _decode_rate(frames_back, key)
            size_kb = os.path.getsize(out_path) / 1024
            results.append((name, hits, n, enc_s, size_kb))
            print(f"[đã xác minh] {name:16s}  {hits:3d}/{n:<3d} decoded   "
                  f"encode {enc_s:5.1f}s   {size_kb:7.1f} KB   frames_out={len(frames_back)}")

    print(f"\nTotal wall time: {time.time() - t0:.1f}s")
    return results


if __name__ == "__main__":
    main()
