"""The ledger (wmapp/ledger.py): the register that turns an ID into a file, a date and a key."""

import multiprocessing
import os
import sqlite3
import sys

import pytest

from wmapp import ledger as L
from wmapp.errors import SpecError
from wmcore import wmid


@pytest.fixture
def src(tmp_path):
    p = tmp_path / "in.png"
    p.write_bytes(b"pretend this is an image")
    return str(p)


@pytest.fixture
def out(tmp_path):
    p = tmp_path / "out.png"
    p.write_bytes(b"the stamped bytes")
    return str(p)


def _reserve(led, src, **kw):
    return led.reserve(kind="image", input_path=src, spec_dict={"text": "@me", "font_path": "/x/y.ttf"}, **kw)


def test_creates_the_file_owner_only_with_a_schema_version(tmp_path):
    path = tmp_path / "sub" / "ledger.sqlite3"
    L.Ledger(path)
    assert path.exists()
    if os.name == "posix":
        assert (path.stat().st_mode & 0o777) == 0o600           # it holds spread-spectrum keys
    con = sqlite3.connect(path)
    assert con.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "1"


def test_reserve_writes_a_pending_row_and_complete_records_the_output_hash(tmp_path, src, out):
    led = L.Ledger(tmp_path / "l.sqlite3")
    wm_id = _reserve(led, src, invisible_mode="spread", ss_key=4242, ss_alpha=12.0)
    assert wmid.parse(wm_id) == wm_id
    rec = led.lookup(wm_id)
    assert rec["status"] == "pending" and rec["kind"] == "image" and rec["input_name"] == "in.png"
    assert rec["output_sha256"] is None and rec["ss_key"] == 4242 and rec["invisible_mode"] == "spread"
    assert "font_path" not in rec["spec"] and rec["spec"]["text"] == "@me"        # machine-local paths are dropped
    led.complete(wm_id, out, ["spread", "metadata"])
    rec = led.lookup(wm_id)
    assert rec["status"] == "done" and rec["layers"] == ["spread", "metadata"] and rec["output_name"] == "out.png"
    assert rec["output_sha256"] == L.file_sha256(out)
    assert led.find_by_output_sha256(L.file_sha256(out))[0]["id"] == wm_id


def test_lookup_accepts_any_spelling_and_returns_none_for_junk_or_unknown(tmp_path, src):
    led = L.Ledger(tmp_path / "l.sqlite3")
    wm_id = _reserve(led, src)
    assert led.lookup(wmid.display(wm_id).lower())["id"] == wm_id
    assert led.lookup("not an id") is None and led.lookup(wmid.generate()) is None


def test_an_explicit_id_is_used_once_only(tmp_path, src):
    led = L.Ledger(tmp_path / "l.sqlite3")
    mine = wmid.generate()
    assert _reserve(led, src, wm_id=mine) == mine
    with pytest.raises(SpecError, match="already recorded"):
        _reserve(led, src, wm_id=mine)
    with pytest.raises(SpecError, match="not a valid watermark ID"):
        _reserve(led, src, wm_id="AAAAAAAAAA")


def test_fail_keeps_the_row_so_an_escaped_id_still_resolves(tmp_path, src):
    led = L.Ledger(tmp_path / "l.sqlite3")
    wm_id = _reserve(led, src)
    led.fail(wm_id, "RuntimeError: boom")
    rec = led.lookup(wm_id)
    assert rec["status"] == "failed" and "boom" in rec["note"]
    assert any("reported a failure" in ln for ln in L.describe(rec))


def test_spread_keys_are_distinct_newest_first_and_spread_only(tmp_path, src):
    led = L.Ledger(tmp_path / "l.sqlite3")
    _reserve(led, src, invisible_mode="spread", ss_key=111)
    _reserve(led, src, invisible_mode="lsb")                       # no key
    _reserve(led, src, invisible_mode="spread", ss_key=222)
    _reserve(led, src, invisible_mode="spread", ss_key=222)        # a batch shares one key: listed once
    assert set(led.spread_keys()) == {111, 222} and len(led.spread_keys()) == 2
    assert led.count() == 4 and len(led.recent(2)) == 2


def test_reading_never_creates_a_ledger(tmp_path):
    missing = tmp_path / "nope.sqlite3"
    led = L.Ledger(missing, create=False)
    assert led.lookup(wmid.generate()) is None and led.spread_keys() == [] and led.count() == 0
    assert not missing.exists()


def test_a_file_that_is_not_a_ledger_is_a_clear_error(tmp_path):
    bad = tmp_path / "bad.sqlite3"
    bad.write_bytes(b"this is not a database" * 50)
    with pytest.raises(SpecError, match="ledger"):
        L.Ledger(bad)
    with pytest.raises(SpecError, match="ledger"):
        L.Ledger(bad, create=False).lookup(wmid.generate())


def test_a_newer_schema_is_refused_rather_than_misread(tmp_path):
    path = tmp_path / "l.sqlite3"
    L.Ledger(path)
    con = sqlite3.connect(path)
    con.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
    con.commit()
    con.close()
    with pytest.raises(SpecError, match="schema version 99"):
        L.Ledger(path)


def test_the_environment_variable_overrides_the_default_path(monkeypatch, tmp_path):
    monkeypatch.setenv(L.ENV_VAR, str(tmp_path / "elsewhere.sqlite3"))
    assert L.default_path() == str(tmp_path / "elsewhere.sqlite3")
    monkeypatch.delenv(L.ENV_VAR)
    assert L.default_path().endswith(os.path.join(".watermark-tool", "ledger.sqlite3"))


def test_public_view_drops_the_key_and_the_settings(tmp_path, src):
    led = L.Ledger(tmp_path / "l.sqlite3")
    rec = led.lookup(_reserve(led, src, invisible_mode="spread", ss_key=999))
    pub = L.public_view(rec)
    assert set(pub) == set(L.PUBLIC_FIELDS) and "ss_key" not in pub and "spec" not in pub
    assert L.public_view(None) is None
    assert not any("999" in ln for ln in L.describe(rec))         # the human report never prints the key


def _worker(args):
    path, src, n = args
    led = L.Ledger(path)
    return [led.reserve(kind="image", input_path=src, spec_dict={}) for _ in range(n)]


def test_processes_writing_at_once_all_get_distinct_ids(tmp_path, src):
    path = str(tmp_path / "l.sqlite3")
    L.Ledger(path)
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(4) as pool:
        results = pool.map(_worker, [(path, src, 15)] * 4)
    ids = [i for chunk in results for i in chunk]
    assert len(ids) == 60 and len(set(ids)) == 60 and L.Ledger(path, create=False).count() == 60
