"""wmapp - the application layer of watermark-tool.

    wmcore   image primitives (grid, blend, LSB / spread-spectrum marks, metadata)
       ^
    wmapp    what the tool DOES: validated requests, the watermark pipeline,
       ^     batch fan-out, read-back, results, notes, limits
       |
    CLI (watermark.py, extract_invisible.py, c2pa_verify.py)   argparse -> spec -> call -> print
    HTTP (serve.py)                                            JSON/bytes -> spec -> call -> JSON/bytes
    Python API                                                 this module, directly

Every front-end is a thin shell that parses its own input format into a
WatermarkSpec / ExtractSpec, calls one function below, and renders the result
its own way. None of them re-implements any behaviour, so they can't drift.

    import sys; sys.path.insert(0, "cli")     # the project isn't pip-installable (yet)
    import wmapp

    r = wmapp.watermark("in.jpg", "out.png", text="@me", blend="multiply",
                        invisible=True, invisible_mode="spread")
    wmapp.extract("out.png", mode="spread", ss_key=r.ss_key).text    # -> "@me"

    r = wmapp.watermark("in.jpg", "out.png", text="@me", stamp_id=True, invisible=True,
                        invisible_mode="spread")     # --id: a short ID in every layer, recorded in the ledger
    wmapp.extract("out.png").record                  # no key needed: the ledger has it
"""

from .errors import LimitError, SpecError, UnavailableError, WmAppError
from .events import Note
from .extraction import ExtractResult, capabilities, extract, verify
from .ledger import Ledger
from .limits import NO_LIMITS, SERVER_LIMITS, Limits, estimate_tiles
from .options import (
    EXTRACT_OPTIONS, OPTIONS, RUN_OPTIONS, SPEC_OPTIONS, ExtractSpec, Option, WatermarkSpec,
    add_arguments, schema, spec_from_namespace,
)
from .pipeline import BatchResult, FileResult, Prepared, prepare, watermark, watermark_batch, watermark_file

__all__ = [
    "watermark", "watermark_file", "watermark_batch", "prepare", "extract", "verify",
    "WatermarkSpec", "ExtractSpec", "Prepared", "FileResult", "BatchResult", "ExtractResult", "Note",
    "Limits", "NO_LIMITS", "SERVER_LIMITS", "estimate_tiles",
    "Ledger", "WmAppError", "SpecError", "UnavailableError", "LimitError",
    "schema", "capabilities", "add_arguments", "spec_from_namespace",
    "OPTIONS", "SPEC_OPTIONS", "RUN_OPTIONS", "EXTRACT_OPTIONS", "Option",
]
