"""wmapp.ledger - the register that turns a short watermark ID into "which file,
when, with which key and settings".

One SQLite file (stdlib sqlite3, no dependency). Default location
~/.watermark-tool/ledger.sqlite3, next to the dev certificate; override with
--ledger PATH or the WATERMARK_TOOL_LEDGER environment variable. Every stamped
output (--id) gets one row:

    id             the watermark ID (wmcore.wmid canonical form) - primary key
    status         'pending' -> 'done' | 'failed'
    created_utc    ISO-8601, UTC
    kind           'image' | 'video'
    input_name / input_sha256      the file that was stamped (its basename, not a path)
    output_name / output_sha256    the stamped result, hashed AFTER every layer
                                   (metadata, C2PA signing) was applied - the hash of
                                   the exact bytes handed out
    invisible_mode, ss_key, ss_alpha   the hidden-mark scheme and its key (NULL if none)
    layers         JSON list of what actually carries the ID: grid, lsb|spread, metadata, c2pa
    spec_json      the request's settings, minus machine-local paths (font, certificate...)

Why the row is written BEFORE the file is stamped ('pending') and completed
after: a crash in between must not leave an image in the wild whose ID the
ledger has never heard of. An ID found in a file always has a row, even if the
run died before it could record the output hash.

What the ledger is not: a proof. It's a private lookup table on your machine.
Anyone who can read an ID out of a file can copy it into another one; what the
ledger's output_sha256 lets `extract` say is whether a suspect file is
byte-identical to what was stamped or merely carries the same ID. The ledger
holds ss_keys - keep the file private (it is created 0600) and don't publish it.

Concurrency: batch workers are separate processes and each writes its own rows;
SQLite serialises them (a 30 s busy timeout, BEGIN IMMEDIATE for the write).
Local disk only - SQLite locking is unreliable over network filesystems.
"""

import hashlib
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from wmcore import wmid

from .errors import SpecError

ENV_VAR = "WATERMARK_TOOL_LEDGER"
DEFAULT_LEDGER_DIR = Path.home() / ".watermark-tool"
SCHEMA_VERSION = 1
_BUSY_TIMEOUT_S = 30
_ID_ATTEMPTS = 20          # fresh IDs to try if a random one is already taken (never happens at 40 bits; cheap insurance)

# Options whose values are paths on the machine that ran the job - useless (and a little revealing) in a record.
_LOCAL_SPEC_KEYS = ("font_path", "face_cascade", "adaptive_preview", "c2pa_cert", "c2pa_key", "ledger")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS watermarks (
    id              TEXT PRIMARY KEY,
    status          TEXT NOT NULL,
    created_utc     TEXT NOT NULL,
    kind            TEXT NOT NULL,
    input_name      TEXT,
    input_sha256    TEXT,
    output_name     TEXT,
    output_sha256   TEXT,
    invisible_mode  TEXT,
    ss_key          INTEGER,
    ss_alpha        REAL,
    layers          TEXT NOT NULL DEFAULT '[]',
    spec_json       TEXT NOT NULL DEFAULT '{}',
    note            TEXT
);
CREATE INDEX IF NOT EXISTS idx_watermarks_output_sha256 ON watermarks(output_sha256);
CREATE INDEX IF NOT EXISTS idx_watermarks_ss_key ON watermarks(ss_key);
"""

_COLUMNS = ("id", "status", "created_utc", "kind", "input_name", "input_sha256", "output_name",
            "output_sha256", "invisible_mode", "ss_key", "ss_alpha", "layers", "spec_json", "note")

# What a record shows to a caller that isn't the ledger's owner (the HTTP adapter): no key, no settings.
PUBLIC_FIELDS = ("id", "status", "created_utc", "kind", "input_name", "output_name", "output_sha256", "layers")


def default_path():
    """Where the ledger lives unless told otherwise: $WATERMARK_TOOL_LEDGER, else ~/.watermark-tool/ledger.sqlite3."""
    return os.environ.get(ENV_VAR) or str(DEFAULT_LEDGER_DIR / "ledger.sqlite3")


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_record(row):
    rec = dict(zip(_COLUMNS, row))
    rec["layers"] = json.loads(rec["layers"] or "[]")
    rec["spec"] = json.loads(rec.pop("spec_json") or "{}")
    return rec


def public_view(record):
    """`record` reduced to PUBLIC_FIELDS (None passes through)."""
    return None if record is None else {k: record.get(k) for k in PUBLIC_FIELDS}


class Ledger:
    """A handle on one ledger file. Cheap to construct; each operation opens and
    closes its own connection, so a Ledger is safe to create in the parent and
    use in worker processes (it carries only the path).

        led = Ledger(path)                  # creates the file (0600) and schema if missing
        led = Ledger(path, create=False)    # read-only use: never creates anything
    """

    def __init__(self, path=None, *, create=True):
        self.path = str(path or default_path())
        self.exists = os.path.exists(self.path)
        if create:
            self._init_schema()
            self.exists = True

    # -- plumbing ----------------------------------------------------------

    def _connect(self):
        try:
            conn = sqlite3.connect(self.path, timeout=_BUSY_TIMEOUT_S, isolation_level=None)
        except sqlite3.Error as exc:
            raise SpecError(f"can't open the ledger at {self.path}: {exc}") from exc
        return conn

    def _init_schema(self):
        parent = os.path.dirname(os.path.abspath(self.path))
        try:
            os.makedirs(parent, exist_ok=True)
            fresh = not os.path.exists(self.path)
            conn = self._connect()
            try:
                conn.executescript(_SCHEMA)
                conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                             (str(SCHEMA_VERSION),))
                row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            finally:
                conn.close()
            if fresh:
                try:
                    os.chmod(self.path, 0o600)      # it holds ss_keys
                except OSError:
                    pass                            # e.g. Windows: best effort
        except (OSError, sqlite3.Error) as exc:
            raise SpecError(f"can't create or open the ledger at {self.path}: {exc}") from exc
        if int(row[0]) > SCHEMA_VERSION:
            raise SpecError(f"the ledger at {self.path} is schema version {row[0]}; this build understands "
                            f"up to {SCHEMA_VERSION} - upgrade watermark-tool")

    def _read(self, sql, params=()):
        if not os.path.exists(self.path):
            return []
        conn = self._connect()
        try:
            return conn.execute(sql, params).fetchall()
        except sqlite3.DatabaseError as exc:
            raise SpecError(f"the file at {self.path} isn't a usable ledger: {exc}") from exc
        finally:
            conn.close()

    # -- writing -----------------------------------------------------------

    def reserve(self, *, kind, input_path, spec_dict, invisible_mode=None, ss_key=None, ss_alpha=None,
                wm_id=None):
        """Allocate an ID and write its 'pending' row; returns the ID. Pass `wm_id`
        to claim a specific one (SpecError if it's malformed or already recorded)."""
        explicit = None
        if wm_id is not None:
            explicit = wmid.parse(wm_id)
            if explicit is None:
                raise SpecError(f"id_value: {wm_id!r} is not a valid watermark ID "
                                f"(10 characters from {wmid.ALPHABET}, with its 2 check characters)")
        input_sha = file_sha256(input_path)
        spec_clean = {k: v for k, v in spec_dict.items() if k not in _LOCAL_SPEC_KEYS and v is not None}
        spec_clean.pop("id_value", None)
        params = [kind, os.path.basename(input_path), input_sha, invisible_mode, ss_key, ss_alpha,
                  json.dumps(spec_clean, sort_keys=True, default=str)]
        conn = self._connect()
        try:
            for _ in range(1 if explicit else _ID_ATTEMPTS):
                candidate = explicit or wmid.generate()
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    conn.execute(
                        "INSERT INTO watermarks(id, status, created_utc, kind, input_name, input_sha256, "
                        "invisible_mode, ss_key, ss_alpha, spec_json) VALUES (?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?)",
                        [candidate, _now()] + params)
                    conn.execute("COMMIT")
                    return candidate
                except sqlite3.IntegrityError:
                    conn.execute("ROLLBACK")
                    if explicit:
                        raise SpecError(f"id_value: {wmid.display(explicit)} is already recorded in the ledger "
                                        f"at {self.path}; every stamped file needs its own ID") from None
                except sqlite3.Error:
                    try:
                        conn.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                    raise
            raise SpecError("couldn't allocate a unique watermark ID")      # unreachable in practice
        finally:
            conn.close()

    def complete(self, wm_id, output_path, layers):
        """Mark `wm_id` done, recording the hash of the final output bytes and which layers carry the ID."""
        self._update(wm_id, "done", output_name=os.path.basename(output_path),
                     output_sha256=file_sha256(output_path), layers=json.dumps(list(layers)))

    def fail(self, wm_id, error):
        """Mark `wm_id` failed (the run raised after reserving it). The row stays: if any partial
        output escaped, its ID is still resolvable."""
        self._update(wm_id, "failed", note=str(error)[:500])

    def _update(self, wm_id, status, **fields):
        cols = ", ".join(f"{k}=?" for k in fields)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(f"UPDATE watermarks SET status=?{', ' + cols if cols else ''} WHERE id=?",
                         [status, *fields.values(), wm_id])
            conn.execute("COMMIT")
        finally:
            conn.close()

    # -- reading -----------------------------------------------------------

    def lookup(self, wm_id):
        """The record for `wm_id` (any accepted spelling), or None."""
        canon = wmid.parse(wm_id)
        if canon is None:
            return None
        rows = self._read(f"SELECT {', '.join(_COLUMNS)} FROM watermarks WHERE id=?", (canon,))
        return _row_to_record(rows[0]) if rows else None

    def find_by_output_sha256(self, sha256):
        rows = self._read(f"SELECT {', '.join(_COLUMNS)} FROM watermarks WHERE output_sha256=? ORDER BY created_utc",
                          (sha256,))
        return [_row_to_record(r) for r in rows]

    def spread_keys(self):
        """Distinct ss_keys of spread-spectrum stamps, newest first - what `extract` tries when you
        don't remember (or never saw) the key. A batch shares one key, so this is one per run, not per file."""
        rows = self._read("SELECT ss_key, MAX(created_utc) AS last FROM watermarks WHERE invisible_mode='spread' "
                          "AND ss_key IS NOT NULL GROUP BY ss_key ORDER BY last DESC, ss_key")
        return [int(r[0]) for r in rows]

    def recent(self, limit=50, status=None):
        sql = f"SELECT {', '.join(_COLUMNS)} FROM watermarks"
        params = []
        if status:
            sql += " WHERE status=?"
            params.append(status)
        sql += " ORDER BY created_utc DESC, rowid DESC LIMIT ?"
        params.append(int(limit))
        return [_row_to_record(r) for r in self._read(sql, params)]

    def count(self):
        rows = self._read("SELECT COUNT(*) FROM watermarks")
        return int(rows[0][0]) if rows else 0


def describe(record, *, exact_match=None):
    """The human-readable lines for a ledger record (shared by extract's report and wm_ledger.py).
    Never includes the ss_key or the settings - those are for the owner to fetch deliberately."""
    when = record.get("created_utc") or "unknown time"
    what = record.get("input_name") or "unknown file"
    lines = [f"Ledger: stamped {when} from '{what}'"
             + (f" as '{record['output_name']}'" if record.get("output_name") else "")
             + f" ({record.get('kind') or 'file'})."]
    layers = record.get("layers") or []
    if layers:
        lines.append("Layers that carry this ID: " + ", ".join(layers) + ".")
    if record.get("status") == "pending":
        lines.append("Note: the run that made this ID never finished recording its output (pending) - "
                     "the file may be a partial or abandoned result.")
    elif record.get("status") == "failed":
        lines.append("Note: the run that made this ID reported a failure - this file may be a partial result.")
    if exact_match is True:
        lines.append("This file is byte-identical to the output recorded for that ID.")
    elif exact_match is False and record.get("output_sha256"):
        lines.append("This file is NOT byte-identical to the recorded output: it was re-encoded or edited after "
                     "stamping - or it is a different image that carries a copied ID. The ID alone can't tell "
                     "those apart.")
    return lines
