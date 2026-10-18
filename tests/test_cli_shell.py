"""The CLIs are now shells over wmapp: what's left to pin down is that the shell itself behaves -
flags come from the schema, errors are argparse usage errors (exit 2), notes are printed as they
always were, exit codes are right. Real subprocesses, as a user would run them."""

import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

import wmapp

CLI = os.path.join(os.path.dirname(__file__), "..", "cli")


def run(script, *args, cwd=None):
    return subprocess.run([sys.executable, os.path.join(CLI, script), *map(str, args)],
                          capture_output=True, text=True, cwd=cwd)


@pytest.fixture()
def img(tmp_path):
    rng = np.random.default_rng(2)
    p = tmp_path / "in.png"
    Image.fromarray(np.clip(rng.normal(120, 40, (240, 320, 3)), 0, 255).astype(np.uint8), "RGB").save(p)
    return str(p)


def test_help_lists_every_schema_flag_in_schema_order():
    out = run("watermark.py", "--help").stdout
    positions = [out.index(o.cli_flag) for o in wmapp.OPTIONS]
    assert positions == sorted(positions), "--help order must follow the schema"
    ex = run("extract_invisible.py", "--help").stdout
    assert all(o.cli_flag in ex for o in wmapp.EXTRACT_OPTIONS)


def test_help_text_comes_from_the_schema():
    out = " ".join(run("watermark.py", "--help").stdout.split())
    assert " ".join(wmapp.OPTIONS[1].help.split())[:40] in out                     # --opacity's help


def test_single_file_prints_saved_and_the_generated_key_note(img, tmp_path):
    out = str(tmp_path / "o.png")
    p = run("watermark.py", img, out, "--text", "x", "--invisible", "--invisible-mode", "spread")
    assert p.returncode == 0 and os.path.exists(out)
    lines = p.stdout.splitlines()
    assert lines[0].startswith("Note: no --ss-key given") and lines[-1] == f"Saved: {out}"


def test_invalid_input_is_an_argparse_usage_error(img, tmp_path):
    for args, fragment in ((["--opacity", "999"], "opacity: must be at most 255"),
                           (["--blend", "screen"], "invalid choice"),
                           (["--c2pa-cert", "/x.pem"], "must be given together"),
                           (["--opacity", "abc"], "invalid int value")):
        p = run("watermark.py", img, str(tmp_path / "o.png"), "--text", "x", *args)
        assert p.returncode == 2 and fragment in p.stderr and p.stderr.startswith("usage:"), args
    assert run("watermark.py", img, str(tmp_path / "o.png")).returncode == 2       # --text is required


def test_batch_mode_prints_the_run_summary(tmp_path, img):
    d = tmp_path / "in"
    d.mkdir()
    for i in range(2):
        Image.open(img).save(d / f"p{i}.png")
    p = run("watermark.py", d, tmp_path / "out", "--text", "x", "--batch", "--workers", "1")
    assert p.returncode == 0
    assert "Watermarking 2 image(s) across 1 worker(s)..." in p.stdout
    assert f"Done: 2/2 image(s) saved to {tmp_path / 'out'}" in p.stdout
    assert sorted(os.listdir(tmp_path / "out")) == ["p0.png", "p1.png"]


def test_extract_exit_codes_and_messages(img, tmp_path):
    out = str(tmp_path / "o.png")
    run("watermark.py", img, out, "--text", "x", "--invisible", "--invisible-text", "secret")
    hit = run("extract_invisible.py", out)
    assert hit.returncode == 0 and hit.stdout.strip() == "Hidden payload: secret"
    miss = run("extract_invisible.py", img, "--mode", "spread", "--ss-key", "5")
    assert miss.returncode == 1 and miss.stdout.startswith("No valid hidden payload found")
    bad = run("extract_invisible.py", out, "--mode", "spread")
    assert bad.returncode == 2 and "--mode spread requires --ss-key" in bad.stderr
