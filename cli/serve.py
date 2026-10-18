#!/usr/bin/env python3
"""
serve.py - the HTTP front-end of watermark-tool: a thin adapter over wmapp
(the application layer), standard library only, so the web page can use the
same pipeline as the CLI and the Python API instead of re-implementing a
subset of it in JavaScript.

    python3 cli/serve.py                 # http://127.0.0.1:8765/  (the web page + the API)
    python3 cli/serve.py --port 9000 --web-dir web

Endpoints (all JSON errors are {"error": {"code": ..., "message": ...}}):

    GET  /                  web/index.html
    GET  /api/schema        every option (type, default, range, help, group), what this
                            install can do (C2PA, EXIF, ffmpeg), and the server's limits -
                            the web page builds its controls from this, so a new option or
                            a changed default in wmapp/options.py shows up with no page edit
    POST /api/watermark     body: the image bytes. header X-WM-Spec: URL-encoded JSON
                              {"spec": {"text": "@me", "blend": "multiply", ...}, "format": "png",
                             "derive_seed_from": "photo1.jpg"}      (optional: per-file seed, as --batch does)
                            -> the watermarked image bytes; header X-WM-Result: JSON
                            {ss_key, wm_id, id_layers, c2pa, kind, seconds, notes: [...]}
    POST /api/extract       body: image bytes, X-WM-Spec: {"spec": {"ss_key": 123, "rescale": true}}
                            -> JSON {found, text, scheme, rescaled_by, confidence, wm_id, record,
                            exact_match, message}. A watermark ID is looked up in THIS server's
                            ledger (--ledger); `record` is cut down to public fields (no key, no settings)
    POST /api/verify        body: file bytes -> the C2PA manifest summary (501 if C2PA isn't installed)

Not a hosting platform. It is meant to run on your own machine (it binds
127.0.0.1 by default) and is hardened enough not to be a hole while it does:

  * requests can't name server-side paths (font, certificate, preview output, ledger...);
    those options exist for the CLI/Python API only (Option.local). --id records into the
    server's own ledger (--ledger, default ~/.watermark-tool/ledger.sqlite3) - a client can
    neither pick the file nor read the spread-spectrum keys in it back,
  * wmapp.SERVER_LIMITS cap upload size, pixel count, grid density and text length,
  * the Host header must be a loopback name or the bound address (DNS-rebinding),
  * no CORS unless you pass --cors-origin ORIGIN, and then only for that origin,
  * at most --max-jobs images are processed at once; the rest get 503.

Images only over HTTP: video is large, slow and already well served by the CLI.
"""

import argparse
import io
import json
import os
import tempfile
import threading
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

import wmapp
import wmcore
from wmcore import wmid
from wmapp import ledger as ledger_mod

DEFAULT_WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")
_FORMATS = {"PNG": ("png", ".png", "image/png"), "JPEG": ("jpeg", ".jpg", "image/jpeg"),
            "WEBP": ("webp", ".webp", "image/webp")}
_ACCEPTED_INPUT = {"PNG", "JPEG", "WEBP", "BMP", "TIFF"}
_LOOPBACK = {"localhost", "127.0.0.1", "[::1]", "::1"}


class _ApiError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _public_notes(notes, scrub):
    """Notes safe to show a client: server paths removed / generalised."""
    out = []
    for n in notes:
        d = n.to_dict()
        if n.code == "c2pa_dev_cert":
            d["message"] = ("Note: signed with this server's local, self-signed dev certificate - the signature is "
                            "real and tamper-evident, but C2PA readers will report the credential as untrusted.")
            d["data"] = {}
        elif n.code == "id_ledger":
            d["message"] = "Note: --id records every output in this server's ledger."
            d["data"] = {}
        elif n.code == "id_stamped":
            carried = ", ".join(n.data.get("layers") or []) or "no layer of this file"
            d["message"] = (f"Watermark ID: {wmid.display(n.data['wm_id'])} (carried by: {carried}); recorded in "
                            f"this server's ledger.")
            d["data"] = {"wm_id": n.data["wm_id"], "layers": list(n.data.get("layers") or [])}
        else:
            d["message"] = d["message"].replace(scrub + os.sep, "")
        out.append(d)
    return out


class _Handler(BaseHTTPRequestHandler):
    server_version = "wmapp"
    protocol_version = "HTTP/1.1"

    # -- plumbing ----------------------------------------------------------

    def log_message(self, fmt, *args):
        if self.server.verbose:
            super().log_message(fmt, *args)

    def _cors(self):
        origin = self.server.cors_origin
        if origin and self.headers.get("Origin") == origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Expose-Headers", "X-WM-Result")
            self.send_header("Vary", "Origin")

    def _send(self, status, body, content_type, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, obj, extra=None):
        self._send(status, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8", extra)

    def _error(self, err):
        self.close_connection = True   # we may not have read the body; don't reuse the connection
        self._json(err.status, {"error": {"code": err.code, "message": err.message}}, {"Connection": "close"})

    def _check_host(self):
        header = self.headers.get("Host") or ""
        if header.startswith("["):                       # IPv6 literal, e.g. [::1]:8765
            host = header.split("]")[0] + "]"
        else:
            host = header.rsplit(":", 1)[0] if ":" in header else header
        if host not in self.server.allowed_hosts:
            raise _ApiError(HTTPStatus.FORBIDDEN, "bad_host", f"Host {host!r} is not allowed (see --allow-host)")

    def _read_body(self):
        raw = self.headers.get("Content-Length")
        if raw is None or not raw.isdigit():
            raise _ApiError(HTTPStatus.LENGTH_REQUIRED, "length_required", "Content-Length is required")
        n = int(raw)
        cap = self.server.limits.max_upload_bytes
        if cap is not None and n > cap:
            raise _ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "too_large",
                            f"upload is {n / 1e6:.1f} MB; this server accepts at most {cap / 1e6:.0f} MB")
        return self.rfile.read(n)

    def _request_json(self):
        raw = self.headers.get("X-WM-Spec")
        if not raw:
            return {}
        try:
            obj = json.loads(urllib.parse.unquote(raw))
        except ValueError:
            raise _ApiError(HTTPStatus.BAD_REQUEST, "bad_request", "X-WM-Spec must be URL-encoded JSON")
        if not isinstance(obj, dict):
            raise _ApiError(HTTPStatus.BAD_REQUEST, "bad_request", "X-WM-Spec must be a JSON object")
        return obj

    # -- routing -----------------------------------------------------------

    def do_OPTIONS(self):   # CORS preflight (only ever answered for the allowed origin)
        try:
            self._check_host()
            if not (self.server.cors_origin and self.headers.get("Origin") == self.server.cors_origin):
                raise _ApiError(HTTPStatus.FORBIDDEN, "cors", "cross-origin requests are not enabled (--cors-origin)")
            self.send_response(HTTPStatus.NO_CONTENT)
            self._cors()
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-WM-Spec")
            self.send_header("Access-Control-Allow-Private-Network", "true")   # public page -> localhost
            self.send_header("Content-Length", "0")
            self.end_headers()
        except _ApiError as err:
            self._error(err)

    def do_GET(self):
        try:
            self._check_host()
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                index = os.path.join(self.server.web_dir, "index.html")
                if not os.path.isfile(index):
                    raise _ApiError(HTTPStatus.NOT_FOUND, "no_web_dir", f"no index.html in {self.server.web_dir}")
                with open(index, "rb") as fh:
                    return self._send(HTTPStatus.OK, fh.read(), "text/html; charset=utf-8")
            if path == "/api/schema":
                body = wmapp.schema(include_local=False)
                lim = self.server.limits
                body["capabilities"] = dict(wmapp.capabilities(), video=False)
                body["limits"] = {"max_upload_bytes": lim.max_upload_bytes, "max_pixels": lim.max_pixels,
                                  "max_tiles": lim.max_tiles, "max_text_chars": lim.max_text_chars}
                body["formats"] = [v[0] for v in _FORMATS.values()]
                return self._json(HTTPStatus.OK, body)
            raise _ApiError(HTTPStatus.NOT_FOUND, "not_found", f"no such resource: {path}")
        except _ApiError as err:
            self._error(err)

    def do_POST(self):
        try:
            self._check_host()
            path = self.path.split("?", 1)[0]
            handler = {"/api/watermark": self._watermark, "/api/extract": self._extract,
                       "/api/verify": self._verify}.get(path)
            if handler is None:
                raise _ApiError(HTTPStatus.NOT_FOUND, "not_found", f"no such resource: {path}")
            request = self._request_json()
            data = self._read_body()
            if not self.server.slots.acquire(timeout=5):
                raise _ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "busy", "the server is busy; try again shortly")
            try:
                handler(request, data)
            finally:
                self.server.slots.release()
        except _ApiError as err:
            self._error(err)
        except wmapp.LimitError as exc:
            self._error(_ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "limit", str(exc)))
        except wmapp.SpecError as exc:
            self._error(_ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_spec", str(exc)))
        except wmapp.UnavailableError as exc:
            self._error(_ApiError(HTTPStatus.NOT_IMPLEMENTED, "unavailable", str(exc)))
        except Exception as exc:    # a bug or a corrupt image: report it, keep the server up
            self._error(_ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "internal", f"{type(exc).__name__}: {exc}"))

    # -- endpoints ---------------------------------------------------------

    def _sniff(self, data):
        try:
            with Image.open(io.BytesIO(data)) as img:
                fmt, size = img.format, img.size
        except (Image.UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
            raise _ApiError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "bad_image", "the body is not a readable image")
        if fmt not in _ACCEPTED_INPUT:
            raise _ApiError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "bad_image", f"unsupported image format {fmt}")
        return fmt, size

    def _watermark(self, request, data):
        spec = wmapp.WatermarkSpec.from_mapping(request.get("spec") or {}, allow_local=False)
        if spec.stamp_id:
            # `ledger` is a local option, so the client couldn't name one; the server's own is used.
            spec = spec.replace(ledger=self.server.ledger or ledger_mod.default_path())
        name = request.get("derive_seed_from")
        if name is not None:
            # What `watermark.py --batch --seed N` does per file: every image gets its own jitter
            # pattern (identical patterns across images are themselves averageable), yet the run
            # stays reproducible. The client passes each file's name; without a seed it's a no-op.
            if not isinstance(name, str):
                raise _ApiError(HTTPStatus.BAD_REQUEST, "bad_request", "derive_seed_from must be a string")
            if spec.seed is not None:
                spec = spec.replace(seed=wmcore.derive_seed(spec.seed, os.path.basename(name)))
        fmt_in, size = self._sniff(data)
        out_key = request.get("format") or {"JPEG": "jpeg", "WEBP": "webp"}.get(fmt_in, "png")
        out = next((v for v in _FORMATS.values() if v[0] == out_key), None)
        if out is None:
            raise _ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_spec", "format must be png, jpeg or webp")
        with tempfile.TemporaryDirectory(prefix="wm-") as tmp:
            src = os.path.join(tmp, "input" + {"JPEG": ".jpg", "WEBP": ".webp", "BMP": ".bmp", "TIFF": ".tif"}.get(fmt_in, ".png"))
            dst = os.path.join(tmp, "output" + out[1])
            with open(src, "wb") as fh:
                fh.write(data)
            result = wmapp.watermark(src, dst, spec, limits=self.server.limits)
            with open(dst, "rb") as fh:
                body = fh.read()
            meta = {"ok": True, "kind": result.kind, "ss_key": result.ss_key, "c2pa": result.c2pa,
                    "wm_id": result.wm_id, "id_layers": result.id_layers,
                    "seconds": round(result.seconds, 3), "format": out[0], "width": size[0], "height": size[1],
                    "notes": _public_notes(result.notes, tmp)}
        self._send(HTTPStatus.OK, body, out[2], {"X-WM-Result": json.dumps(meta)})

    def _extract(self, request, data):
        spec = wmapp.ExtractSpec.from_mapping(request.get("spec") or {}, allow_local=False)
        if self.server.ledger:      # this server's own ledger - or none, if nothing has been stamped into it yet
            spec = (spec.replace(ledger=self.server.ledger) if os.path.exists(self.server.ledger)
                    else spec.replace(use_ledger=False))
        self._sniff(data)
        with tempfile.TemporaryDirectory(prefix="wm-") as tmp:
            src = os.path.join(tmp, "input")
            with open(src, "wb") as fh:
                fh.write(data)
            result = wmapp.extract(src, spec)
        self._json(HTTPStatus.OK, result.to_dict(public=True))

    def _verify(self, request, data):
        with tempfile.TemporaryDirectory(prefix="wm-") as tmp:
            fmt, _ = self._sniff(data)
            src = os.path.join(tmp, "input" + _FORMATS.get(fmt, ("", ".png", ""))[1])
            with open(src, "wb") as fh:
                fh.write(data)
            info = wmapp.verify(src)
        self._json(HTTPStatus.OK, info)


def make_server(host="127.0.0.1", port=8765, web_dir=None, limits=wmapp.SERVER_LIMITS, cors_origin=None,
                allow_hosts=(), max_jobs=2, verbose=False, ledger=None):
    """Build (not start) the server; port=0 picks a free port (see server.server_port).
    `ledger` is the ledger file --id requests record into (default: the standard one)."""
    srv = ThreadingHTTPServer((host, port), _Handler)
    srv.daemon_threads = True
    srv.web_dir = os.path.abspath(web_dir or DEFAULT_WEB_DIR)
    srv.limits, srv.cors_origin, srv.verbose = limits, cors_origin, verbose
    srv.ledger = ledger
    srv.slots = threading.BoundedSemaphore(max_jobs)
    bound = f"[{host}]" if ":" in host else host
    srv.allowed_hosts = set(_LOOPBACK) | {bound} | set(allow_hosts)
    return srv


def main():
    p = argparse.ArgumentParser(description="Serve the web tool and the wmapp HTTP API (images only).")
    p.add_argument("--host", default="127.0.0.1", help="Address to bind (default: 127.0.0.1 - this machine only)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--web-dir", default=DEFAULT_WEB_DIR, help="Directory containing index.html")
    p.add_argument("--cors-origin", default=None, help="Allow this ONE origin to call the API from a browser "
                   "(e.g. https://you.github.io); default: same-origin only")
    p.add_argument("--allow-host", action="append", default=[], help="Extra Host header value to accept (repeatable)")
    p.add_argument("--max-jobs", type=int, default=2, help="Images processed concurrently (default: 2)")
    p.add_argument("--ledger", default=None, help="Ledger file that --id requests record into and /api/extract "
                   "looks IDs up in (default: $WATERMARK_TOOL_LEDGER or ~/.watermark-tool/ledger.sqlite3)")
    p.add_argument("--verbose", action="store_true", help="Log each request")
    args = p.parse_args()
    srv = make_server(args.host, args.port, args.web_dir, cors_origin=args.cors_origin,
                      allow_hosts=args.allow_host, max_jobs=args.max_jobs, verbose=args.verbose,
                      ledger=args.ledger)
    print(f"Serving http://{args.host}:{srv.server_port}/  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
