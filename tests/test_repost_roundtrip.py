"""bench/repost_roundtrip.py: the bookkeeping for the real-platform test. The 'platform' here is a LOCAL STAND-IN (a resize and a
JPEG re-encode) - it checks that the harness reads files back and reports honestly, not how any real platform behaves."""

import importlib.util
import io
import json
import os

import numpy as np
import pytest
from PIL import Image

_spec = importlib.util.spec_from_file_location("repost_roundtrip", os.path.join(os.path.dirname(__file__), "..", "bench", "repost_roundtrip.py"))
rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rr)


def _photo(path, seed, w=800, h=600):
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.linspace(0, 255, w), np.linspace(0, 255, h))
    tex = np.kron(rng.normal(0, 14, (-(-h // 8), -(-w // 8), 1)), np.ones((8, 8, 1)))[:h, :w]
    arr = np.clip(np.stack([xx, yy, (xx + yy) / 2], -1) + tex + rng.normal(0, 6, (h, w, 3)), 0, 255)
    Image.fromarray(arr.astype(np.uint8)).save(path)
    return str(path)


def test_prepare_then_check_reads_back_a_standin_repost_and_reports_each_file(tmp_path):
    srcs = [_photo(tmp_path / f"p{i}.png", 70 + i) for i in range(2)]
    out = tmp_path / "rt"
    m = rr.prepare(out, srcs, jpeg=True)
    assert len(m["files"]) == 4 and (out / "manifest.json").exists() and (out / "ledger.sqlite3").exists()
    ids = {e["file"]: e["wm_id"] for e in m["files"]}
    assert ids["p0_stamped.png"] == ids["p0_stamped_q95.jpg"] != ids["p1_stamped.png"]

    down = tmp_path / "down"
    down.mkdir()
    for e in m["files"]:                                           # the stand-in 'platform': max side 640 px + JPEG q75, renamed
        img = Image.open(out / "upload" / e["file"]).convert("RGB")
        img.thumbnail((640, 640), Image.LANCZOS)
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=75)
        (down / ("dl_" + os.path.splitext(e["file"])[0] + ".jpg")).write_bytes(buf.getvalue())
    (down / "notes.txt").write_text("not an image")                # ignored
    (down / "unrelated.jpg").write_bytes(open(srcs[0], "rb").read() and _jpeg_bytes(_photo(tmp_path / "other.png", 5)))

    rows, unmatched = rr.check(out, down, rescale=True, budget=30)
    assert len(rows) == 5
    by = {r["file"]: r for r in rows}
    assert not by["unrelated.jpg"]["read"] and not by["unrelated.jpg"].get("matches")        # an image with no mark: honestly "no"
    for r in rows:
        assert set(r) >= {"file", "bytes", "seconds", "read", "layers"}
    ok = [r for r in rows if r.get("matches")]
    assert ok, "the resized + JPEG q75 copies were not read at all"                          # at least some read back through --rescale
    for r in ok:
        assert r["layers"].count(":") == 4 and "spread:ID" in r["layers"]                    # four layers reported, spread read it
    assert isinstance(unmatched, list)
    rr._print(rows, unmatched)                                                               # the table prints without error


def _jpeg_bytes(path):
    buf = io.BytesIO(); Image.open(path).convert("RGB").save(buf, "JPEG", quality=90); return buf.getvalue()


def test_a_file_the_tool_cannot_open_is_reported_not_raised(tmp_path):
    src = _photo(tmp_path / "p.png", 3)
    out = tmp_path / "rt"
    rr.prepare(out, [src])
    down = tmp_path / "down"; down.mkdir()
    (down / "broken.jpg").write_bytes(b"\xff\xd8\xff not really a jpeg")
    rows, unmatched = rr.check(out, down)
    assert len(rows) == 1 and rows[0]["read"] is False and unmatched == ["p_stamped.png"]
