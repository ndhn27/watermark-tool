"""The point of the application layer, as a test: the same job run through the CLI (a real
subprocess), the Python API and the HTTP server yields the SAME pixels - and reading the hidden
mark back through each front-end yields the same answer. If any front-end grows its own idea of
a default, a unit, or a code path, this is where it shows."""

import io
import json
import os
import subprocess
import sys
import threading
import urllib.parse
import urllib.request

import numpy as np
import pytest
from PIL import Image

import serve
import wmapp

CLI = os.path.join(os.path.dirname(__file__), "..", "cli")
KEY = 424242


def _source(tmp_path, w=640, h=480, seed=11):
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.linspace(0, 255, w), np.linspace(0, 255, h))
    arr = np.clip(np.stack([xx, yy, (xx + yy) / 2], -1) + rng.normal(0, 10, (h, w, 3)), 0, 255).astype(np.uint8)
    p = tmp_path / "in.png"
    Image.fromarray(arr, "RGB").save(p)
    return str(p)


def _px(data_or_path):
    src = io.BytesIO(data_or_path) if isinstance(data_or_path, bytes) else data_or_path
    return np.asarray(Image.open(src).convert("RGBA"))


def _cli(script, *args, cwd=None):
    return subprocess.run([sys.executable, os.path.join(CLI, script), *map(str, args)],
                          capture_output=True, text=True, cwd=cwd)


@pytest.fixture(scope="module")
def http(tmp_path_factory):
    # The server's own ledger (a client can't choose one): a module-scoped file the --id parity test uses.
    srv = serve.make_server(port=0, ledger=str(tmp_path_factory.mktemp("srv") / "ledger.sqlite3"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _http_watermark(srv, src_path, spec, fmt="png"):
    with open(src_path, "rb") as fh:
        body = fh.read()
    r = urllib.request.Request(f"http://127.0.0.1:{srv.server_port}/api/watermark", data=body, method="POST")
    r.add_header("X-WM-Spec", urllib.parse.quote(json.dumps({"spec": spec, "format": fmt})))
    with urllib.request.urlopen(r) as resp:
        return resp.read(), json.loads(resp.headers["X-WM-Result"])


def _http_extract(srv, data, spec):
    r = urllib.request.Request(f"http://127.0.0.1:{srv.server_port}/api/extract", data=data, method="POST")
    r.add_header("X-WM-Spec", urllib.parse.quote(json.dumps({"spec": spec})))
    with urllib.request.urlopen(r) as resp:
        return json.loads(resp.read())


# One job exercising the visible grid, a blend mode, jitter seed, an invisible spread-spectrum mark and metadata.
JOB = dict(text="@parity", seed=7, blend="multiply", ink=0.4, opacity=100, spacing=180, font_size=28, angle=25,
           invisible=True, invisible_mode="spread", ss_key=KEY, ss_alpha=13.0, author="Jane", copyright_text="CC")
CLI_ARGS = ["--text", "@parity", "--seed", "7", "--blend", "multiply", "--ink", "0.4", "--opacity", "100",
            "--spacing", "180", "--font-size", "28", "--angle", "25", "--invisible", "--invisible-mode", "spread",
            "--ss-key", str(KEY), "--ss-alpha", "13.0", "--author", "Jane", "--copyright", "CC"]


def test_cli_python_api_and_http_produce_identical_pixels(tmp_path, http):
    src = _source(tmp_path)
    cli_out, api_out = str(tmp_path / "cli.png"), str(tmp_path / "api.png")

    proc = _cli("watermark.py", src, cli_out, *CLI_ARGS)
    assert proc.returncode == 0, proc.stderr
    wmapp.watermark(src, api_out, **JOB)
    http_bytes, meta = _http_watermark(http, src, JOB)

    a, b, c = _px(cli_out), _px(api_out), _px(http_bytes)
    assert a.shape == (480, 640, 4)
    assert np.array_equal(a, b), "CLI and Python API diverged"
    assert np.array_equal(a, c), "HTTP and CLI diverged"
    assert meta["ss_key"] == KEY


def test_the_hidden_mark_reads_back_the_same_through_every_front_end(tmp_path, http):
    src = _source(tmp_path)
    out = str(tmp_path / "out.png")
    assert _cli("watermark.py", src, out, *CLI_ARGS).returncode == 0

    cli = _cli("extract_invisible.py", out, "--ss-key", KEY)
    api = wmapp.extract(out, ss_key=KEY)
    with open(out, "rb") as fh:
        over_http = _http_extract(http, fh.read(), {"ss_key": KEY})

    assert cli.returncode == 0 and cli.stdout.splitlines() == api.report_lines() == over_http["message"].split("\n")
    assert api.text == over_http["text"] == "@parity"

    wrong_cli = _cli("extract_invisible.py", out, "--ss-key", KEY + 1)
    wrong_api = wmapp.extract(out, ss_key=KEY + 1)
    with open(out, "rb") as fh:
        wrong_http = _http_extract(http, fh.read(), {"ss_key": KEY + 1})
    assert wrong_cli.returncode == 1 and not wrong_api.found and wrong_http["found"] is False
    assert wrong_cli.stdout.splitlines() == wrong_api.report_lines()


def test_the_per_layer_report_is_the_same_through_every_front_end(tmp_path, http):
    # --layers / layers=True / {"layers": true}: one option row, so one report. And the structured per-layer data
    # (what a script or the web page reads) is in the JSON whether or not the text asked for it.
    src = _source(tmp_path)
    out = str(tmp_path / "out.png")
    assert _cli("watermark.py", src, out, *CLI_ARGS).returncode == 0

    cli = _cli("extract_invisible.py", out, "--ss-key", KEY, "--layers")
    api = wmapp.extract(out, ss_key=KEY, layers=True)
    with open(out, "rb") as fh:
        data = fh.read()
    over_http = _http_extract(http, data, {"ss_key": KEY, "layers": True})
    assert cli.returncode == 0, cli.stdout + cli.stderr
    assert cli.stdout.splitlines() == api.report_lines() == over_http["message"].split("\n")
    assert any("layer by layer" in ln for ln in api.report_lines())

    quiet = _http_extract(http, data, {"ss_key": KEY})              # not asked for: text unchanged, data still there
    assert "layer by layer" not in quiet["message"]
    assert quiet["layers"] == over_http["layers"] == [lr.to_dict() for lr in api.layers]
    assert [lr["layer"] for lr in quiet["layers"]] == ["lsb", "spread", "metadata", "c2pa"]


def test_the_three_front_ends_agree_on_defaults(tmp_path, http):
    # Nothing but the text (and a seed - unseeded output is random by design, so it can't be compared):
    # whatever each front-end thinks "default" means must be the same thing.
    src = _source(tmp_path)
    cli_out, api_out = str(tmp_path / "cli.png"), str(tmp_path / "api.png")
    assert _cli("watermark.py", src, cli_out, "--text", "hello", "--seed", "3").returncode == 0
    wmapp.watermark(src, api_out, text="hello", seed=3)
    http_bytes, _ = _http_watermark(http, src, {"text": "hello", "seed": 3})
    assert np.array_equal(_px(cli_out), _px(api_out)) and np.array_equal(_px(cli_out), _px(http_bytes))


def test_the_three_front_ends_refuse_the_same_bad_input_with_the_same_words(tmp_path, http):
    src = _source(tmp_path)
    cli = _cli("watermark.py", src, str(tmp_path / "o.png"), "--text", "x", "--opacity", "999")
    assert cli.returncode == 2 and "opacity: must be at most 255, got 999" in cli.stderr
    with pytest.raises(wmapp.SpecError, match="opacity: must be at most 255, got 999"):
        wmapp.watermark(src, str(tmp_path / "o.png"), text="x", opacity=999)
    with pytest.raises(urllib.error.HTTPError) as e:
        _http_watermark(http, src, {"text": "x", "opacity": 999})
    assert e.value.code == 422 and "opacity: must be at most 255, got 999" in json.loads(e.value.read())["error"]["message"]


# --- --id: the same job stamps the same pixels and reads back the same report through every front-end ------------

def _id_job():
    import random
    from wmcore import wmid
    wm_id = wmid.generate(random.Random(99))
    return wm_id, dict(JOB, stamp_id=True, grid_id=True, id_value=wm_id), CLI_ARGS + ["--id", "--grid-id", "--id-value", wm_id]


def test_stamping_an_id_gives_identical_pixels_and_the_same_read_back_everywhere(tmp_path, http):
    wm_id, job, args = _id_job()
    src = _source(tmp_path)
    cli_out, api_out = str(tmp_path / "cli.png"), str(tmp_path / "api.png")
    cli_ledger, api_ledger = str(tmp_path / "cli.sqlite3"), str(tmp_path / "api.sqlite3")

    proc = _cli("watermark.py", src, cli_out, *args, "--ledger", cli_ledger)
    assert proc.returncode == 0, proc.stderr
    r = wmapp.watermark(src, api_out, ledger=api_ledger, **job)
    http_bytes, meta = _http_watermark(http, src, job)          # the server records into ITS ledger (client can't choose)

    assert r.wm_id == meta["wm_id"] == wm_id
    assert r.id_layers == meta["id_layers"] == ["grid", "spread", "metadata"]
    cli_px, api_px, http_px = _px(cli_out), _px(api_out), _px(http_bytes)
    assert np.array_equal(cli_px, api_px), "CLI and Python API diverged"
    assert np.array_equal(cli_px, http_px), "HTTP and CLI diverged"

    # No key given to anyone: every front-end finds it through its own ledger and leads with the same words.
    cli = _cli("extract_invisible.py", cli_out, "--ledger", cli_ledger)
    api = wmapp.extract(api_out, ledger=api_ledger)
    over_http = _http_extract(http, http_bytes, {})
    assert cli.returncode == 0, cli.stdout + cli.stderr
    assert api.wm_id == over_http["wm_id"] == wm_id and api.scheme == over_http["scheme"] == "spread"
    lead = api.report_lines()[:2]
    assert cli.stdout.splitlines()[:2] == over_http["message"].split("\n")[:2] == lead
    assert api.exact_match is True and over_http["exact_match"] is True

    # What HTTP will not say: the key, the settings, or where the server keeps its ledger.
    assert over_http["record"]["id"] == wm_id and "ss_key" not in over_http["record"] and "spec" not in over_http["record"]
    assert "ledger" not in over_http and str(tmp_path) not in json.dumps(over_http)
    assert api.record["ss_key"] == KEY                          # ...which the owner, via the API, can still get
