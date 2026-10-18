#!/usr/bin/env python3
"""P4 (bước 3/5) - Follow-up của video_invisible_probe.py: ở CRF23 (H.264 mặc định của x264),
alpha=12.0 (DEFAULT_SS_ALPHA hiện tại, chỉnh cho JPEG ảnh) chỉ đọc được ~65-78% khung hình
(xem p4-video-invisible-probe.txt). Script này dò xem tăng alpha có đóng khoảng cách đó không,
và cái giá phải trả là gì (mean|Δpixel| - đổi màu trung bình mỗi pixel, một proxy thô cho độ
"lộ" của dấu, không phải phép đo tri giác thật).

    python3 bench/experiments/video_alpha_sweep.py
"""

import os, sys, subprocess, tempfile, time
sys.path.insert(0, os.path.abspath("cli"))
sys.path.insert(0, os.path.abspath("bench"))
sys.path.insert(0, os.path.abspath("bench/experiments"))
import numpy as np, cv2
from PIL import Image
import bench_corpus as C
import wmcore
from wmcore import wmid

SEED = 20260922 + 4_000_000
W, H, FPS, N = 640, 480, 24, 60

def make_pan(seed, w, h, n, pad=160):
    big = np.asarray(C._pink(w+pad, h+pad, seed).convert("RGB"), dtype=np.uint8)
    out = []
    for i in range(n):
        t = i / max(1, n-1)
        off = int(round(pad*0.5*(1-np.cos(2*np.pi*t))))
        out.append(np.ascontiguousarray(big[off:off+h, off:off+w]))
    return out

def run_ffmpeg(args, frames, w, h, fps):
    cmd = ["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{w}x{h}","-r",str(fps),"-i","-",*args]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in frames: p.stdin.write(np.ascontiguousarray(f,dtype=np.uint8).tobytes())
    p.stdin.close(); err=p.stderr.read(); rc=p.wait()
    if rc!=0: raise RuntimeError(err.decode(errors="replace")[:500])

def read_frames(path):
    cap = cv2.VideoCapture(path); out=[]
    while True:
        ok, f = cap.read()
        if not ok: break
        out.append(f[...,::-1])
    cap.release(); return out

def decode_rate(frames, key):
    hits=0
    for f in frames:
        img = Image.fromarray(f, "RGB")
        corrs = wmcore.spread_spectrum_block_correlations(img, key)
        if wmcore.find_spread_spectrum_id(corrs) is not None: hits+=1
    return hits, len(frames)

wm_id = wmid.generate()
key = wmcore.derive_seed(SEED, "alpha_sweep")
src = make_pan(SEED, W, H, N)

with tempfile.TemporaryDirectory() as tmp:
    for alpha in (12.0, 18.0, 24.0, 32.0, 40.0):
        wm_frames = [np.asarray(wmcore.embed_spread_spectrum(Image.fromarray(f,"RGB").convert("RGBA"), wm_id, key, alpha=alpha).convert("RGB"), dtype=np.uint8) for f in src]
        # quality proxy: mean abs pixel delta vs source
        delta = np.mean([np.abs(wf.astype(np.int16)-sf.astype(np.int16)).mean() for wf,sf in zip(wm_frames, src)])
        out_path = os.path.join(tmp, f"a{int(alpha)}.mp4")
        run_ffmpeg(["-c:v","libx264","-preset","medium","-crf","23","-pix_fmt","yuv420p", out_path], wm_frames, W, H, FPS)
        frames_back = read_frames(out_path)
        hits, n = decode_rate(frames_back, key)
        print(f"[đã xác minh] alpha={alpha:5.1f}  crf23  {hits:3d}/{n:<3d}  mean|Δpixel|={delta:.3f}")
