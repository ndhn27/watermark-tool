"""Repost round trip: the one test nothing in this repository can run for you - post a stamped file to a REAL platform
(YouTube, Facebook, Telegram, TikTok, X, Instagram ...), download it back, and see what the tool still reads.

The platforms recompress in their own ways (resize to their own maximum, re-encode with their own settings, strip metadata,
convert video to their own ladder), so every number in bench/results/ measured with a local JPEG / x264 / ffmpeg stand-in is a
PROXY until this has been run. This script only does the bookkeeping, so that the run is cheap and the result is one table.

    # 1. stamp what you will upload (your own photos / videos; ID + spread-spectrum mark + ledger, as in normal use)
    python3 bench/repost_roundtrip.py prepare --out /tmp/rt photo1.jpg photo2.png clip.mp4 [--jpeg] [--c2pa]
    #    -> /tmp/rt/upload/*   (post THESE; keep the file names or not - the ID inside identifies each one)
    #    -> /tmp/rt/manifest.json + /tmp/rt/ledger.sqlite3

    # 2. post them on each platform, download the posted copies back into one folder PER PLATFORM
    #    (right-click / the platform's own download; a screenshot or a re-saved copy is a different test: say so in the folder name)

    # 3. read them back
    python3 bench/repost_roundtrip.py check --out /tmp/rt --downloads /tmp/rt_downloads/telegram [--rescale] [--json r.json]

`check` prints one row per downloaded file: size, whether an ID was read and whether it is one of the IDs you uploaded (and
which file), which layers (LSB / spread-spectrum / metadata / C2PA) each found on their own, whether the copy looks cropped or
resized, and how long it took. It also lists the uploaded files that no downloaded copy matched.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cli"))
import wmapp                                                   # noqa: E402
import wmcore                                                  # noqa: E402
from wmcore import wmid                                        # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = set(wmcore.VIDEO_EXTS) if hasattr(wmcore, "VIDEO_EXTS") else {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare(out, inputs, *, text="@repost-test", jpeg=False, c2pa=False, workers=2):
    """Stamp each input into <out>/upload/ and write <out>/manifest.json. Returns the manifest (a dict)."""
    out = Path(out)
    upload = out / "upload"
    upload.mkdir(parents=True, exist_ok=True)
    ledger = str(out / "ledger.sqlite3")
    entries = []
    for src in inputs:
        src = Path(src)
        kind = "video" if src.suffix.lower() in VIDEO_EXTS else "image"
        dst = upload / (src.stem + ("_stamped.mp4" if kind == "video" else "_stamped.png"))
        kwargs = dict(text=text, stamp_id=True, ledger=ledger, invisible=True, invisible_mode="spread")
        if c2pa:
            kwargs["c2pa"] = True
        if kind == "video":
            kwargs["workers"] = workers
        t0 = time.perf_counter()
        r = wmapp.watermark(str(src), str(dst), **kwargs)
        made = [dst]
        if jpeg and kind == "image":
            from PIL import Image
            j = upload / (src.stem + "_stamped_q95.jpg")
            Image.open(dst).convert("RGB").save(j, quality=95)
            made.append(j)
        for path in made:
            entries.append({"file": path.name, "source": str(src), "kind": kind, "wm_id": r.wm_id,
                            "bytes": path.stat().st_size, "sha256": _sha256(path), "stamp_seconds": round(time.perf_counter() - t0, 1)})
        print(f"stamped {src.name} -> {', '.join(p.name for p in made)}  ID {wmid.display(r.wm_id)}")
    manifest = {"ledger": ledger, "upload_dir": str(upload), "files": entries}
    with open(out / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n{len(entries)} file(s) in {upload}\nPost them, download the posted copies into a folder per platform, then:\n"
          f"  python3 bench/repost_roundtrip.py check --out {out} --downloads <folder>")
    return manifest


def _layer_summary(res):
    short = {"found": "ID", "text": "text", "none": "-", "inconclusive": "?", "skipped": "n/a"}
    names = {"lsb": "LSB", "spread": "spread", "metadata": "meta", "c2pa": "C2PA"}
    return " ".join(f"{names[lr.layer]}:{short[lr.status]}" for lr in res.layers)


def check(out, downloads, *, rescale=False, budget=40.0, only_ext=None):
    """Read every image/video under `downloads` against the ledger of <out>; return a list of row dicts."""
    out = Path(out)
    with open(out / "manifest.json", encoding="utf-8") as fh:
        manifest = json.load(fh)
    by_id = {}
    for e in manifest["files"]:
        by_id.setdefault(wmid.parse(e["wm_id"]) or e["wm_id"], []).append(e["file"])
    rows = []
    for path in sorted(Path(downloads).rglob("*")):
        ext = path.suffix.lower()
        if not path.is_file() or ext not in (IMAGE_EXTS | VIDEO_EXTS) or (only_ext and ext not in only_ext):
            continue
        t0 = time.perf_counter()
        try:
            res = wmapp.extract(str(path), ledger=manifest["ledger"], rescale=rescale and ext in IMAGE_EXTS,
                                rescale_budget_seconds=budget)
            err = None
        except Exception as exc:                                # a platform's odd container: report it, keep going
            res, err = None, f"{type(exc).__name__}: {str(exc)[:80]}"
        row = {"file": str(path.relative_to(downloads)), "bytes": path.stat().st_size, "seconds": round(time.perf_counter() - t0, 1)}
        if res is None:
            row.update(read=False, matches=None, layers="", note=err)
        else:
            known = by_id.get(res.wm_id) if res.wm_id else None
            note = []
            if res.regrid is not None:
                note.append("looks cropped")
            if res.rescaled_by is not None:
                note.append(f"resized (restored by x{res.rescaled_by:.3f})")
            if res.flipped:
                note.append(f"mirrored ({res.flipped})")
            if res.video_scan is not None:
                note.append(f"video: {res.video_scan.hits}/{res.video_scan.frames_checked} frames")
            if res.budget_exhausted or res.resync_budget_exhausted:
                note.append("search ran out of time (inconclusive)")
            row.update(read=bool(res.wm_id), wm_id=res.wm_id, matches=known, scheme=res.scheme,
                       layers=_layer_summary(res), note="; ".join(note))
        rows.append(row)
    seen = {name for r in rows for name in (r.get("matches") or [])}
    unmatched = [e["file"] for e in manifest["files"] if e["file"] not in seen]
    return rows, unmatched


def _print(rows, unmatched):
    if not rows:
        print("no image or video files found in the downloads folder")
        return
    w = max(len(r["file"]) for r in rows)
    print(f"{'downloaded file':{w}s} | {'MB':>6s} | ID read | which upload      | layers (each on its own)               | s    | notes")
    for r in rows:
        which = ("+".join(r["matches"]) if r.get("matches") else ("UNKNOWN ID" if r.get("read") else "-"))
        print(f"{r['file']:{w}s} | {r['bytes']/1e6:6.2f} | {'yes' if r['read'] else 'NO ':7s} | {which:17.17s} | {r['layers']:38s} | {r['seconds']:4.1f} | {r.get('note') or ''}")
    n_ok = sum(1 for r in rows if r.get("matches"))
    print(f"\n{n_ok} of {len(rows)} downloaded file(s) read as one of the IDs you uploaded.")
    if unmatched:
        print("uploaded files that no downloaded copy matched: " + ", ".join(unmatched))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="stamp files to upload")
    p.add_argument("--out", required=True); p.add_argument("inputs", nargs="+")
    p.add_argument("--text", default="@repost-test"); p.add_argument("--jpeg", action="store_true", help="also write a q95 JPEG of each image")
    p.add_argument("--c2pa", action="store_true", help="also sign a C2PA manifest (needs c2pa-python 0.37.3-0.37.10)")
    c = sub.add_parser("check", help="read downloaded copies back")
    c.add_argument("--out", required=True); c.add_argument("--downloads", required=True)
    c.add_argument("--rescale", action="store_true", help="also search for a resize / crop+resize on images (slower)")
    c.add_argument("--budget", type=float, default=40.0); c.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.out, a.inputs, text=a.text, jpeg=a.jpeg, c2pa=a.c2pa)
    else:
        rows, unmatched = check(a.out, a.downloads, rescale=a.rescale, budget=a.budget)
        _print(rows, unmatched)
        if a.json:
            with open(a.json, "w", encoding="utf-8") as fh:
                json.dump({"rows": rows, "unmatched_uploads": unmatched}, fh, indent=2)


if __name__ == "__main__":
    main()
