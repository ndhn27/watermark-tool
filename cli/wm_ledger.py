#!/usr/bin/env python3
"""
wm_ledger.py - look things up in (and check on) the watermark ledger that
`watermark.py --id` writes.

    python3 wm_ledger.py show K3M9-QXR2-T4        # everything the ledger knows about one ID
    python3 wm_ledger.py show K3M9QXR2T4 --secrets   # ...including the spread-spectrum key and settings
    python3 wm_ledger.py list                     # newest first (-n 50, --status pending)
    python3 wm_ledger.py find suspect.jpg         # is this exact file (by SHA-256) one we stamped?
    python3 wm_ledger.py keys                     # spread-spectrum keys on record (what extract tries)

The ledger is $WATERMARK_TOOL_LEDGER, else ~/.watermark-tool/ledger.sqlite3 (or
--ledger PATH). It is only ever read here. The IDs are found in a suspect
image with extract_invisible.py; this is the look-it-up half for when you
already have one - typed from a visible grid (--grid-id), a video's C2PA
manifest (c2pa_verify.py prints it) or metadata.
"""

import argparse
import json
import os
import sys

import wmapp
from wmapp import ledger as ledger_mod
from wmcore import wmid


def _open(path):
    path = path or ledger_mod.default_path()
    if not os.path.exists(path):
        sys.exit(f"No ledger at {path} (nothing has been stamped with --id there yet, or pass --ledger PATH).")
    try:
        return ledger_mod.Ledger(path, create=False)
    except wmapp.SpecError as exc:
        sys.exit(str(exc))


def _summary_line(rec):
    return (f"{wmid.display(rec['id'])}  {rec['created_utc']}  {rec['status']:<7}  {rec['kind']:<5}  "
            f"{rec.get('input_name') or '?'} -> {rec.get('output_name') or '?'}  [{', '.join(rec['layers'])}]")


def main(argv=None):
    p = argparse.ArgumentParser(description="Look up watermark IDs in the ledger written by watermark.py --id.")
    p.add_argument("--ledger", help="Ledger file (default: $WATERMARK_TOOL_LEDGER or ~/.watermark-tool/ledger.sqlite3)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("show", help="Print one ID's record")
    sp.add_argument("id", help="A watermark ID, with or without dashes, any case")
    sp.add_argument("--secrets", action="store_true", help="Also print the spread-spectrum key and the stamping settings")
    sp.add_argument("--json", action="store_true", help="Print the full record as JSON (includes the key and settings)")
    lp = sub.add_parser("list", help="Newest records first")
    lp.add_argument("-n", type=int, default=50, help="How many (default 50)")
    lp.add_argument("--status", choices=("pending", "done", "failed"))
    fp = sub.add_parser("find", help="Look a file up by its SHA-256")
    fp.add_argument("file")
    sub.add_parser("keys", help="Spread-spectrum keys on record, newest first")
    args = p.parse_args(argv)

    led = _open(args.ledger)

    if args.cmd == "show":
        rec = led.lookup(args.id)
        if rec is None:
            print(f"{args.id!r} is not in the ledger at {led.path}"
                  + ("" if wmid.parse(args.id) else " (and it isn't a well-formed ID - check the characters)") + ".")
            return 1
        if args.json:
            print(json.dumps(rec, indent=2, sort_keys=True))
            return 0
        print(f"Watermark ID: {wmid.display(rec['id'])}")
        for line in ledger_mod.describe(rec):
            print(line)
        if rec.get("output_sha256"):
            print(f"Output SHA-256: {rec['output_sha256']}")
        if args.secrets:
            print(f"Hidden mark: {rec.get('invisible_mode') or 'none'}"
                  + (f", ss_key {rec['ss_key']}, ss_alpha {rec['ss_alpha']}" if rec.get("ss_key") is not None else ""))
            print("Settings: " + json.dumps(rec["spec"], sort_keys=True))
        return 0

    if args.cmd == "list":
        rows = led.recent(args.n, args.status)
        for rec in rows:
            print(_summary_line(rec))
        print(f"({len(rows)} shown, {led.count()} in {led.path})")
        return 0

    if args.cmd == "find":
        matches = led.find_by_output_sha256(ledger_mod.file_sha256(args.file))
        if not matches:
            print("No ledger record has this exact file as its output (it may still be a re-encoded copy - "
                  "run extract_invisible.py on it).")
            return 1
        for rec in matches:
            print(_summary_line(rec))
        return 0

    if args.cmd == "keys":
        keys = led.spread_keys()
        for k in keys:
            print(k)
        print(f"({len(keys)} distinct key(s))", file=sys.stderr)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
