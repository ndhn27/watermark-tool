"""The HTTP adapter (cli/serve.py), against a real server on an ephemeral port: the API works,
and - just as important for something that takes input from a network - what it refuses."""

import io
import json
import threading
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pytest
from PIL import Image

import serve
import wmapp


def _png(w=400, h=300, seed=1):
    rng = np.random.default_rng(seed)
    arr = np.clip(rng.normal(120, 40, (h, w, 3)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def server():
    srv = serve.make_server(port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def call(srv, path, body=None, req=None, headers=None):
    r = urllib.request.Request(f"http://127.0.0.1:{srv.server_port}{path}", data=body,
                               method="POST" if body is not None else "GET")
    if req is not None:
        r.add_header("X-WM-Spec", urllib.parse.quote(json.dumps(req)))
    for k, v in (headers or {}).items():
        r.add_header(k, v)
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers), err.read()


def error_of(srv, path, body, req, **kw):
    status, _, raw = call(srv, path, body, req, **kw)
    return status, json.loads(raw)["error"]


# --- the happy path ---------------------------------------------------------

def test_index_and_schema(server):
    status, headers, body = call(server, "/")
    assert status == 200 and headers["Content-Type"].startswith("text/html") and b'id="wm-server"' in body
    status, _, raw = call(server, "/api/schema")
    schema = json.loads(raw)
    assert status == 200 and schema["limits"]["max_pixels"] == wmapp.SERVER_LIMITS.max_pixels
    assert schema["capabilities"]["video"] is False and set(schema["capabilities"]) >= {"c2pa", "exif", "ffmpeg"}
    names = {o["name"] for o in schema["watermark"]}
    assert {"blend", "ss_key", "invisible"} <= names
    assert not {"font_path", "c2pa_cert", "c2pa_key", "face_cascade", "adaptive_preview", "batch", "workers"} & names


def test_watermark_returns_image_bytes_and_a_result_header(server):
    status, headers, body = call(server, "/api/watermark", _png(),
                                 {"spec": {"text": "@me", "seed": 7, "invisible": True, "invisible_mode": "spread"}})
    assert status == 200 and headers["Content-Type"] == "image/png"
    assert Image.open(io.BytesIO(body)).size == (400, 300)
    meta = json.loads(headers["X-WM-Result"])
    assert meta["ok"] and meta["kind"] == "image" and isinstance(meta["ss_key"], int) and meta["format"] == "png"
    assert [n["code"] for n in meta["notes"]] == ["ss_key_generated"]


def test_a_round_trip_over_http_recovers_the_hidden_text(server):
    _, headers, marked = call(server, "/api/watermark", _png(seed=3),
                              {"spec": {"text": "@me", "invisible": True, "invisible_mode": "spread", "ss_key": 4242}})
    status, _, raw = call(server, "/api/extract", marked, {"spec": {"ss_key": 4242}})
    out = json.loads(raw)
    assert status == 200 and out["found"] and out["text"] == "@me" and out["scheme"] == "spread"
    status, _, raw = call(server, "/api/extract", marked, {"spec": {"ss_key": 1}})
    assert status == 200 and json.loads(raw)["found"] is False


def test_output_format_can_be_chosen(server):
    _, headers, body = call(server, "/api/watermark", _png(), {"spec": {"text": "x"}, "format": "jpeg"})
    assert headers["Content-Type"] == "image/jpeg" and Image.open(io.BytesIO(body)).format == "JPEG"
    assert json.loads(headers["X-WM-Result"])["format"] == "jpeg"


def test_notes_do_not_leak_server_paths(server):
    _, headers, _ = call(server, "/api/watermark", _png(), {"spec": {"text": "x", "invisible": True}, "format": "jpeg"})
    notes = json.loads(headers["X-WM-Result"])["notes"]
    assert [n["code"] for n in notes] == ["lsb_lossy_format"]
    assert "wm-" not in notes[0]["message"] and "/tmp" not in notes[0]["message"]


def test_derive_seed_from_matches_what_batch_mode_does(server):
    same = call(server, "/api/watermark", _png(), {"spec": {"text": "x", "seed": 7}})[2]
    a = call(server, "/api/watermark", _png(), {"spec": {"text": "x", "seed": 7}, "derive_seed_from": "a.png"})[2]
    b = call(server, "/api/watermark", _png(), {"spec": {"text": "x", "seed": 7}, "derive_seed_from": "b.png"})[2]
    assert same != a != b
    # and it is exactly wmcore.derive_seed(seed, name): the same pixels as asking for that seed directly
    import wmcore
    direct = call(server, "/api/watermark", _png(), {"spec": {"text": "x", "seed": wmcore.derive_seed(7, "a.png")}})[2]
    assert direct == a


# --- what it refuses --------------------------------------------------------

@pytest.mark.parametrize("spec, status, code", [
    ({"text": "x", "font_path": "/etc/passwd"}, 422, "invalid_spec"),        # server paths are not for clients
    ({"text": "x", "c2pa": True, "c2pa_cert": "/x.pem", "c2pa_key": "/x.key"}, 422, "invalid_spec"),
    ({"text": "x", "adaptive": True, "adaptive_preview": "/tmp/pwned.png"}, 422, "invalid_spec"),
    ({"text": "x", "opacity": 999}, 422, "invalid_spec"),
    ({"text": "x", "blnd": "multiply"}, 422, "invalid_spec"),
    ({"text": "x", "batch": True}, 422, "invalid_spec"),
    ({"text": "x", "spacing": 2}, 413, "limit"),                              # ~160k tiles: refused before any work
    ({"text": "x" * 500}, 413, "limit"),
    ({"opacity": 10}, 422, "invalid_spec"),                                    # text missing
])
def test_bad_or_abusive_specs_are_refused(server, spec, status, code):
    got_status, err = error_of(server, "/api/watermark", _png(), {"spec": spec})
    assert (got_status, err["code"]) == (status, code), err


def test_dense_grid_is_refused_quickly_not_computed(server):
    import time
    t0 = time.perf_counter()
    error_of(server, "/api/watermark", _png(2000, 1500), {"spec": {"text": "x", "spacing": 1}})
    assert time.perf_counter() - t0 < 5


def test_garbage_and_wrong_types_are_415(server):
    assert error_of(server, "/api/watermark", b"definitely not an image", {"spec": {"text": "x"}})[0] == 415
    assert error_of(server, "/api/extract", b"\x00" * 64, {"spec": {}})[0] == 415


def test_malformed_spec_header_is_400(server):
    r = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/api/watermark", data=_png(), method="POST")
    r.add_header("X-WM-Spec", "%7Bnot json")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(r)
    assert e.value.code == 400


def test_unknown_routes_404(server):
    assert call(server, "/etc/passwd")[0] == 404
    assert call(server, "/api/nope", b"x")[0] == 404
    assert call(server, "/../cli/serve.py")[0] == 404            # only index.html is ever served


def test_a_foreign_host_header_is_rejected_against_dns_rebinding(server):
    status, _, raw = call(server, "/api/schema", headers={"Host": "evil.example"})
    assert status == 403 and json.loads(raw)["error"]["code"] == "bad_host"
    assert call(server, "/api/schema", headers={"Host": "localhost:8765"})[0] == 200


def test_oversized_uploads_are_refused_without_reading_them():
    small = wmapp.Limits(max_upload_bytes=1000, max_pixels=40_000_000, max_tiles=60_000, max_text_chars=200)
    srv = serve.make_server(port=0, limits=small)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        status, err = error_of(srv, "/api/watermark", _png(), {"spec": {"text": "x"}})
        assert status == 413 and err["code"] == "too_large"
    finally:
        srv.shutdown()
        srv.server_close()


def test_verify_reports_501_when_c2pa_is_unavailable(server, monkeypatch):
    import c2pa_provenance
    monkeypatch.setattr(c2pa_provenance, "AVAILABLE", False)
    status, err = error_of(server, "/api/verify", _png(), {})
    assert status == 501 and err["code"] == "unavailable" and "c2pa-python" in err["message"]


# --- CORS -------------------------------------------------------------------

def test_no_cors_headers_by_default(server):
    _, headers, _ = call(server, "/api/schema", headers={"Origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin" not in headers


def test_cors_is_opt_in_and_for_exactly_one_origin():
    srv = serve.make_server(port=0, cors_origin="https://me.github.io")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        _, ok, _ = call(srv, "/api/schema", headers={"Origin": "https://me.github.io"})
        assert ok["Access-Control-Allow-Origin"] == "https://me.github.io" and ok["Access-Control-Expose-Headers"] == "X-WM-Result"
        _, other, _ = call(srv, "/api/schema", headers={"Origin": "https://evil.example"})
        assert "Access-Control-Allow-Origin" not in other
        pre = urllib.request.Request(f"http://127.0.0.1:{srv.server_port}/api/watermark", method="OPTIONS")
        pre.add_header("Origin", "https://me.github.io")
        with urllib.request.urlopen(pre) as r:
            assert r.status == 204 and "X-WM-Spec" in r.headers["Access-Control-Allow-Headers"]
            assert r.headers["Access-Control-Allow-Private-Network"] == "true"
        bad = urllib.request.Request(f"http://127.0.0.1:{srv.server_port}/api/watermark", method="OPTIONS")
        bad.add_header("Origin", "https://evil.example")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(bad)
        assert e.value.code == 403
    finally:
        srv.shutdown()
        srv.server_close()


def test_the_server_stays_up_after_errors(server):
    for _ in range(3):
        call(server, "/api/watermark", b"garbage", {"spec": {"text": "x"}})
    assert call(server, "/api/schema")[0] == 200


# --- --id over HTTP: the server's ledger, never the client's -----------------------------------------------------

@pytest.fixture
def id_server(tmp_path):
    srv = serve.make_server(port=0, ledger=str(tmp_path / "server-ledger.sqlite3"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def test_id_requests_record_into_the_servers_ledger_and_hide_its_path(id_server, tmp_path):
    spec = {"text": "x", "seed": 1, "stamp_id": True, "invisible": True, "invisible_mode": "spread", "ss_key": 77}
    status, headers, body = call(id_server, "/api/watermark", _png(640, 480), {"spec": spec, "format": "png"})
    assert status == 200
    meta = json.loads(headers["X-WM-Result"])
    assert meta["wm_id"] and meta["id_layers"] == ["spread", "metadata"]
    assert str(tmp_path) not in json.dumps(meta)                      # the ledger path never reaches the client
    assert [n["code"] for n in meta["notes"] if n["code"].startswith("id_")] == ["id_ledger", "id_no_hidden_mark"][:1] + ["id_stamped"]
    ledger = wmapp.Ledger(str(tmp_path / "server-ledger.sqlite3"), create=False)
    assert ledger.lookup(meta["wm_id"])["ss_key"] == 77
    status, _, raw = call(id_server, "/api/extract", body, {"spec": {}})
    found = json.loads(raw)
    assert status == 200 and found["wm_id"] == meta["wm_id"] and found["exact_match"] is True
    assert "ss_key" not in found["record"] and "ledger" not in found


def test_a_client_cannot_name_a_ledger_file(id_server, tmp_path):
    status, err = error_of(id_server, "/api/watermark", _png(), {"spec": {"text": "x", "stamp_id": True,
                                                                          "ledger": str(tmp_path / "mine.sqlite3")}})
    assert status == 422 and "ledger" in err["message"] and not (tmp_path / "mine.sqlite3").exists()
    status, err = error_of(id_server, "/api/extract", _png(), {"spec": {"ledger": "/etc/passwd"}})
    assert status == 422 and "ledger" in err["message"]


def test_schema_marks_the_ledger_local_and_hides_it_from_clients(server):
    _, _, raw = call(server, "/api/schema")
    names = {o["name"] for o in json.loads(raw)["watermark"]}
    assert {"stamp_id", "id_value", "grid_id"} <= names and "ledger" not in names
    assert "ledger" not in {o["name"] for o in json.loads(raw)["extract"]}
