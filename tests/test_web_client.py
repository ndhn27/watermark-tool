"""Runs the JavaScript tests (tests/js/) for web/index.html's server mode against a REAL serve.py
started here. Needs Node >= 20 (skipped without it). wm-server.test.mjs needs nothing else;
page-smoke.test.mjs boots the whole page in jsdom and skips itself unless `npm install` was run in
tests/js."""

import glob
import os
import shutil
import subprocess
import threading

import pytest

import serve

JS_DIR = os.path.join(os.path.dirname(__file__), "js")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_web_client_against_a_real_server():
    srv = serve.make_server(port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        tests = sorted(glob.glob(os.path.join(JS_DIR, "*.test.mjs")))
        env = dict(os.environ, WM_TEST_SERVER=f"http://127.0.0.1:{srv.server_port}")
        proc = subprocess.run(["node", "--test", *tests], capture_output=True, text=True, env=env, timeout=180)
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1500:]
        assert "# fail 0" in proc.stdout
    finally:
        srv.shutdown()
        srv.server_close()
