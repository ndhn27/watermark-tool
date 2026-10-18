"""Real-footage video check for crop resync: OpenCV's vtest.avi (768x576, 10 fps, real pedestrians), the first N frames,
optionally scaled up to 1080p (real content, blurrier than native 1080p). Stamped through the production pipeline, then
cropped / re-encoded with ffmpeg + x264, then read back with `wmapp.extract`.

    python3 bench/experiments/real_video_probe.py --src /tmp/realdata/vtest.avi --frames 60
    python3 bench/experiments/real_video_probe.py --src /tmp/realdata/vtest.avi --frames 24 --size 1920x1080
"""
import argparse, os, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "cli"))
import wmapp


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--size", default=None, help="WxH to scale the footage to first (e.g. 1920x1080)")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    d = tempfile.mkdtemp()
    src = os.path.join(d, "src.mp4")
    vf = f"scale={a.size.replace('x', ':')}:flags=lanczos" if a.size else "null"
    ff("-i", a.src, "-vf", vf, "-frames:v", str(a.frames), "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", src)
    t = time.perf_counter()
    out = os.path.join(d, "out.mp4")
    r = wmapp.watermark(src, out, text="@real", stamp_id=True, invisible=True, invisible_mode="spread", workers=a.workers)
    dt = time.perf_counter() - t
    w, h = (int(v) for v in (a.size or "768x576").split("x"))
    print(f"footage {w}x{h}, {a.frames} frames; stamped in {dt:.0f}s ({dt/a.frames:.2f}s/frame, {a.workers} workers); ID {r.wm_id}")
    c = 80 if w < 1000 else 200
    cases = [("no crop (control), CRF23", "null", 23),
             (f"trim {c}px off both sides, CRF23", f"crop={w-2*c}:{h}:{c}:0", 23),
             ("9:16 centre strip, CRF23", f"crop={(h*9//16)//2*2}:{h}:{(w-(h*9//16))//4*2}:0", 23),
             ("off-grid crop 37,23 (-50,-31), CRF23", f"crop={w-50-37}:{h-31-23}:37:23", 23),
             ("off-grid crop, CRF28", f"crop={w-50-37}:{h-31-23}:37:23", 28)]
    print(f"{'attack':40s} | resync OFF: frames read | resync ON: frames read (via re-alignment) | extract time ON")
    for name, vfilter, crf in cases:
        p = os.path.join(d, "a.mp4")
        ff("-i", out, "-vf", vfilter, "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", p)
        off = wmapp.extract(p, ss_key=r.ss_key, use_ledger=False, resync=False).video_scan
        t = time.perf_counter(); on = wmapp.extract(p, ss_key=r.ss_key, use_ledger=False).video_scan; dt = time.perf_counter() - t
        print(f"{name:40s} | {off.hits:3d}/{off.frames_checked:<3d}                  | {on.hits:3d}/{on.frames_checked:<3d} ({on.regrid_hits:3d})                       | {dt:5.1f}s", flush=True)


if __name__ == "__main__":
    main()
