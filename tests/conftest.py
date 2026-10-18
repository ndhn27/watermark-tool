import os
import sys

# wmcore.py etc. live in cli/ and import each other with plain top-level
# imports (`import wmcore`), not package-relative ones - see wmcore.py's
# module docstring for why (multiprocessing picklability). Tests need the
# same sys.path setup a `python3 cli/watermark.py ...` invocation gets for
# free (Python adds the script's own directory to sys.path).
CLI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cli"))
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

# bench/ follows the same flat-import convention as cli/ (see bench/README.md).
BENCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bench"))
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path, monkeypatch):
    """No test may read or write the developer's real ~/.watermark-tool/ledger.sqlite3: point the default
    ledger at a per-test path that doesn't exist yet. It reaches subprocess CLIs through the environment.
    (Tests that want a ledger pass one explicitly, or create this one by using --id.)"""
    monkeypatch.setenv("WATERMARK_TOOL_LEDGER", str(tmp_path / "default-ledger.sqlite3"))
