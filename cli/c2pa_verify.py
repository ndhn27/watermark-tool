#!/usr/bin/env python3
"""
c2pa_verify.py - Read back the C2PA (Content Credentials) manifest embedded
by `watermark.py --c2pa`, alongside extract_invisible.py for the pixel-domain
invisible marks. See cli/c2pa_provenance.py's module docstring for how
signing works, what the auto-generated dev certificate does and doesn't
prove, and why "signing certificate untrusted" is the expected result until
you supply your own --c2pa-cert/--c2pa-key.

Usage:
    python3 c2pa_verify.py watermarked.png
    python3 c2pa_verify.py watermarked.mp4 --json
"""

import argparse
import json
import sys

import wmapp


def _print_report(info, path):
    if not info["has_manifest"]:
        print(f"No C2PA manifest found in {path} (never signed with --c2pa, or it was stripped/re-encoded since).")
        return 1

    print(f"Title: {info.get('title')}")
    if info.get("claim_generator"):
        print(f"Claim generator: {info['claim_generator']}")
    if info.get("ingredients"):
        print(f"Ingredient(s): {', '.join(info['ingredients'])}")
    if info.get("actions"):
        print(f"Actions: {', '.join(info['actions'])}")
    if info.get("author"):
        print(f"Author (CreativeWork assertion): {info['author']}")
    if info.get("wm_id"):
        print(f"Watermark ID (CreativeWork assertion): {info['wm_id']}  "
              f"(look it up: python3 cli/wm_ledger.py show {info['wm_id']})")

    print(f"Validation state: {info.get('validation_state')}")
    results = info.get("validation_results") or {}
    active = results.get("activeManifest", {})
    for failure in active.get("failure", []):
        code = failure.get("code")
        note = ""
        if code == "signingCredential.untrusted":
            note = " (expected for the auto-generated dev cert - see cli/c2pa_provenance.py / README.md)"
        print(f"  - FAILED {code}: {failure.get('explanation')}{note}")
    for info_item in active.get("informational", []):
        print(f"  - note: {info_item.get('code')}: {info_item.get('explanation')}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Read back a C2PA manifest embedded by watermark.py --c2pa.")
    parser.add_argument("file", help="Path to a file stamped with watermark.py --c2pa")
    parser.add_argument("--json", action="store_true", help="Print the full parsed summary as JSON instead of a human-readable report")
    args = parser.parse_args()

    try:
        info = wmapp.verify(args.file)
    except wmapp.UnavailableError as exc:
        print(exc)
        sys.exit(1)

    if args.json:
        print(json.dumps(info, indent=2))
        sys.exit(0 if info["has_manifest"] else 1)

    sys.exit(_print_report(info, args.file))


if __name__ == "__main__":
    main()
