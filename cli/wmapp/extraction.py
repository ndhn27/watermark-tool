"""wmapp.extraction - read-back use-cases: pull a hidden payload out of an
image, and read a C2PA manifest. The decoding itself lives in wmcore
(invisible.py, spread.py, rescale.py); this layer owns the *policy* -
which schemes to try in which order, what counts as inconclusive - and the
wording of the outcome, so every front-end reports it the same way.

A payload that is a well-formed watermark ID (wmcore.wmid: 10 characters and
their check characters) is treated differently from free text: it is looked
up in the ledger (wmapp/ledger.py), the spread-spectrum keys the ledger
recorded are tried when no --ss-key is given, and a real ID is preferred over
a junk LSB decode or a legacy text payload. Free-text payloads (the tool's
behaviour before --id existed) read back exactly as they always did.

Every layer is checked on its own (v26): in "auto" mode extract no longer stops at the first layer that
yields an ID, it reads LSB, spread-spectrum, metadata and C2PA and records each one's outcome in
ExtractResult.layers (a LayerResult per layer - found / text / none / inconclusive / skipped, never folded
into one number). The headline fields (found, text, scheme, ...) are chosen exactly as before, so existing
callers see no change; layers are what a caller reads to learn which layers of a reposted copy survived.
"""

import os
import shutil
import time
from dataclasses import dataclass, field, replace as _dc_replace
from typing import Optional

from PIL import Image

import c2pa_provenance
import video as video_mod
import wmcore
from wmcore import wmid
from wmcore.invisible import (
    extract_invisible, extract_spread_spectrum_oriented, extract_spread_spectrum_rescaled,
    iter_spread_spectrum_payloads_oriented,
)
from wmcore.orient import ORIENTATION_LABELS, ORIENTATIONS

from . import ledger as ledger_mod
from .errors import SpecError, UnavailableError
from .options import ExtractSpec

_SCHEME_NAMES = {"lsb": "LSB mark", "spread": "spread-spectrum mark", "metadata": "image metadata",
                 "c2pa": "C2PA manifest"}


_LAYER_ORDER = ("lsb", "spread", "metadata", "c2pa")     # also the precedence of the headline result


@dataclass
class LayerResult:
    """What ONE layer found on its own. `status`:
        "found"        a well-formed watermark ID was read (wm_id set, canonical form)
        "text"         a payload was read that is not a watermark ID: a free-text payload from before --id
                       existed - or junk (an unmarked image / a wrong key can yield a short one; this tool
                       cannot tell the two apart)
        "none"         the layer was checked and nothing readable was there
        "inconclusive" the layer's search ran out of time before covering its whole space (not a confirmed absence)
        "skipped"      the layer was not checked - `note` says why (a mode, a missing key, a missing package)
    `note` is a short human-readable qualifier (how the mark was recovered, how many keys/frames, why skipped)."""
    layer: str                              # "lsb" | "spread" | "metadata" | "c2pa"
    status: str
    text: Optional[str] = None
    wm_id: Optional[str] = None
    note: Optional[str] = None

    def describe(self):
        note = f" ({self.note})" if self.note else ""
        if self.status == "found":
            return f"found {wmid.display(self.wm_id)}{note}"
        if self.status == "text":
            shown = self.text if len(self.text) <= 40 else self.text[:37] + "..."
            return f"read a payload that is not a watermark ID: {shown!r}{note}"
        if self.status == "inconclusive":
            return f"ran out of time before finishing - not a confirmed absence{note}"
        if self.status == "skipped":
            return f"not checked{note}"
        return f"nothing readable{note}"

    def to_dict(self):
        return {"layer": self.layer, "status": self.status, "text": self.text, "wm_id": self.wm_id,
                "note": self.note}


def _read(layer, text, note=None):
    """A layer's outcome from the payload it decoded (None = nothing)."""
    if text is None:
        return LayerResult(layer, "none", note=note)
    canon = wmid.parse(text)
    if canon:
        return LayerResult(layer, "found", text=canon, wm_id=canon, note=note)
    return LayerResult(layer, "text", text=text, note=note)


def _skipped(layer, why):
    return LayerResult(layer, "skipped", note=why)


@dataclass
class ExtractResult:
    found: bool
    text: Optional[str] = None              # the payload; for a watermark ID, its canonical form
    scheme: Optional[str] = None            # "lsb" | "spread" | "metadata" | "c2pa"
    rescaled_by: Optional[float] = None     # rescale search: the decode-time resize factor that worked
    confidence: Optional[float] = None      # rescale search only; a heuristic index, not a probability
    budget_exhausted: bool = False          # "not found" that is really "ran out of time"
    ss_key_supplied: bool = False           # (only used for wording the not-found hint)
    rescale_tried: bool = False
    flipped: Optional[str] = None           # spread only: the copy was mirrored - "h" (left-right), "v" (top-bottom), "hv"
    regrid: Optional[tuple] = None          # spread only: (dy, dx), where the 8x8 block grid starts in the copy - set
                                             # only when the ID was read after re-aligning it, i.e. the copy looks CROPPED
    resync_tried: bool = False              # the crop re-alignment search ran (it is on by default)
    resync_budget_exhausted: bool = False   # ...and ran out of time: a "not found" that is really "ran out of time"
    scaled_keys_reached: Optional[int] = None  # --rescale: the crop + resize search ran out of time after sweeping this many
                                             # ledger keys (so a long ledger's older stamps were never looked at)
    wm_id: Optional[str] = None             # set when `text` is a well-formed watermark ID
    record: Optional[dict] = None           # that ID's ledger record, if a ledger was consulted and knows it
    exact_match: Optional[bool] = None      # suspect file's SHA-256 == the ledger's output hash (None: unknown)
    ledger_path: Optional[str] = None       # the ledger that was consulted (None: none was)
    keys_available: int = 0                 # spread keys the ledger offered (0 when --ss-key was given)
    keys_tried: int = 0                     # ...and how many of them were actually decoded with
    ledger_error: Optional[str] = None      # the DEFAULT ledger exists but couldn't be read (extract went on without it)
    video_scan: Optional[object] = None     # video only: the VideoScanResult for the key that was used
                                             # (found or not) - the full per-frame breakdown, not just found/text
    layers: list = field(default_factory=list)  # one LayerResult per layer, in _LAYER_ORDER, always all four -
                                             # what each layer found ON ITS OWN, whichever one the headline above used
    show_layers: bool = False               # --layers: put the per-layer lines in report_lines() too

    def report_lines(self):
        """The human-readable outcome, exactly as extract_invisible.py prints it."""
        lines = self._outcome_lines()
        lines.extend(self._disagreement_lines())
        if self.show_layers:
            lines.extend(self.layer_lines())
        if self.ledger_error:
            lines.append(f"Note: the default ledger couldn't be read, so it was not used ({self.ledger_error}).")
        return lines

    @property
    def disagreement(self):
        """{ID: [layers that read it]} when two or more layers read DIFFERENT valid IDs, else {}. The headline
        picks one of them (first in LSB, spread, metadata, C2PA order) - this is how the others are not lost."""
        ids = {}
        for lr in self.layers:
            if lr.status == "found":
                ids.setdefault(lr.wm_id, []).append(lr.layer)
        return ids if len(ids) > 1 else {}

    def _disagreement_lines(self):
        ids = self.disagreement
        if not ids:
            return []
        parts = "; ".join(f"{' + '.join(_SCHEME_NAMES[x] for x in layers)} read {wmid.display(i)}"
                          for i, layers in ids.items())
        return [f"Warning: the layers disagree - {parts}. A file stamped once carries the same ID in every layer, "
                "so this copy was probably stamped again after the first time, or one layer misread. The result "
                "above is the first match in the order LSB, spread-spectrum, metadata, C2PA - not necessarily "
                "the original stamp."]

    def layer_lines(self):
        """One line per layer: what that layer found on its own (empty if the result has no layer data)."""
        if not self.layers:
            return []
        width = max(len(_SCHEME_NAMES[lr.layer]) for lr in self.layers) + 1
        return (["This file, layer by layer (each checked on its own - a hit in one says nothing about the others):"]
                + [f"  {_SCHEME_NAMES[lr.layer] + ':':<{width}} {lr.describe()}" for lr in self.layers])

    def _outcome_lines(self):
        if not self.found:
            if self.video_scan is not None:
                scan = self.video_scan
                return [f"No watermark ID decoded from {scan.frames_checked} checked frame(s) of the video "
                        f"(0/{scan.frames_checked} hit) - the file may never have been stamped with "
                        f"--invisible --invisible-mode spread, or the --ss-key/--jitter-refresh-seconds given "
                        f"doesn't match what embedded it. A handful of missed frames near a real scene cut is "
                        f"expected even on a correctly-keyed video (see README, 'Invisible mark' under Video "
                        f"support) - this is 0 hits across the whole scan, not a near-miss."]
            if self.budget_exhausted or self.resync_budget_exhausted:
                tried = (f" (reached {self.keys_tried} of {self.keys_available} ledger keys)"
                         if self.keys_available and not self.resync_budget_exhausted else "")
                if self.scaled_keys_reached is not None and self.scaled_keys_reached < (self.keys_available or 0):
                    tried += (f" (the crop + resize search got through {self.scaled_keys_reached} of "
                              f"{self.keys_available} ledger keys, newest first)")
                bigger = ("--resync-budget-seconds" if self.resync_budget_exhausted else "--rescale-budget-seconds")
                return [f"No hidden payload found before the search's time budget ran out{tried} - this is "
                        f"inconclusive, not a confirmed absence. Try again with a larger {bigger}."]
            if self.keys_available:
                hint = (f"tried the {self.keys_tried} spread-spectrum key(s) in the ledger"
                        + ("" if self.rescale_tried else "; if it was resized after stamping, try --rescale"))
            elif not self.ss_key_supplied:
                hint = "pass --ss-key if it was embedded with --invisible-mode spread"
            else:
                hint = "if it was resized after stamping, try --rescale"
            return ["No valid hidden payload found (image may be recompressed, resized, cropped, or never "
                    f"stamped; {hint})."]
        if self.wm_id:
            via = _SCHEME_NAMES.get(self.scheme, self.scheme)
            if self.scheme == "spread" and self.keys_available:
                via += ", with a key from the ledger"
            lines = [f"Watermark ID: {wmid.display(self.wm_id)}", f"(found in the {via})"]
            if self.scheme in ("metadata", "c2pa"):
                lines.append("(that is the weakest layer: metadata and manifests are stripped or edited by "
                             "almost every platform, so a match here says the file was passed on intact, "
                             "not that a pixel mark survived)")
        else:
            lines = [f"Hidden payload: {self.text}"]
        if self.rescaled_by is not None:
            lines.append(f"(recovered after resampling the image by x{self.rescaled_by:.4f}, i.e. it had been resized "
                         f"to about {1 / self.rescaled_by:.1%} of its original size)")
        if self.flipped:
            lines.append(f"(recovered after flipping the image back: this copy had been {ORIENTATION_LABELS[self.flipped]})")
        if self.regrid is not None:
            lines.append("(recovered after re-aligning to the 8x8 pixel grid the mark was stamped on: this copy looks "
                         "cropped - a left/right trim, or an offset that is not a multiple of 8 pixels"
                         + (", and it was resized too" if self.rescaled_by is not None else "") + ")")
        if self.video_scan is not None and self.video_scan.regrid_hits:
            lines.append(f"(video: {self.video_scan.regrid_hits} of the {self.video_scan.hits} frame(s) that decoded needed the "
                         "crop re-alignment: this video looks cropped)")
        if self.video_scan is not None:
            scan = self.video_scan
            lines.append(f"(video: {scan.hits}/{scan.frames_checked} frame(s) decoded this ID; a video gives one "
                         f"decode attempt per frame, so this is the fraction that hit, not a confidence score - "
                         f"see README, 'Invisible mark' under Video support)")
        if self.confidence is not None:
            lines.append(f"(match confidence: {self.confidence:.2f} - a heuristic index, not a calibrated probability; "
                         "see wmcore/rescale.py's module docstring)")
        if self.wm_id and self.record:
            lines.extend(ledger_mod.describe(self.record, exact_match=self.exact_match))
        elif self.wm_id and self.ledger_path:
            lines.append(f"That ID is not in the ledger at {self.ledger_path}: it was stamped somewhere else, "
                         "the ID was misread, or it is a copy of an ID that was never yours.")
        return lines

    def to_dict(self, *, public=False):
        """JSON-able form. public=True is what the HTTP adapter serves: the ledger record is cut down
        to ledger.PUBLIC_FIELDS (no key, no settings) and no server-side path is named."""
        record = ledger_mod.public_view(self.record) if public else self.record
        d = {"found": self.found, "text": self.text, "scheme": self.scheme, "rescaled_by": self.rescaled_by,
             "confidence": self.confidence, "budget_exhausted": self.budget_exhausted, "flipped": self.flipped,
             "cropped": self.regrid is not None, "resync_budget_exhausted": self.resync_budget_exhausted,
             "scaled_keys_reached": self.scaled_keys_reached,
             "wm_id": self.wm_id, "record": record, "exact_match": self.exact_match,
             "keys_available": self.keys_available, "keys_tried": self.keys_tried,
             "video": self.video_scan.to_dict() if self.video_scan is not None else None,
             "layers": [lr.to_dict() for lr in self.layers], "layers_disagree": bool(self.disagreement)}
        view = _dc_replace(self, ledger_error="this server's ledger file is unreadable") if (
            public and self.ledger_error) else self
        lines = view.report_lines()
        if public:
            if self.ledger_path:
                lines = [ln.replace(self.ledger_path, "the server's ledger") for ln in lines]
        else:
            d["ledger"] = self.ledger_path
        d["message"] = "\n".join(lines)
        return d


def _open_ledger(spec):
    """(ledger, error): the ledger to consult, or None. An explicit --ledger that doesn't exist or isn't
    a ledger is an error (a typo should not read as 'no record'); the default location is used only if
    it already exists, extract never creates one, and a default ledger that turns out to be unreadable
    is set aside with an explanation rather than making every extraction fail."""
    if not spec.use_ledger:
        return None, None
    if spec.ledger:
        if not os.path.exists(spec.ledger):
            raise SpecError(f"ledger: no such file: {spec.ledger}")
        led = ledger_mod.Ledger(spec.ledger, create=False)
        led.count()                                     # SpecError if it isn't a usable ledger
        return led, None
    path = ledger_mod.default_path()
    if not os.path.exists(path):
        return None, None
    led = ledger_mod.Ledger(path, create=False)
    try:
        led.count()
    except SpecError as exc:
        return None, str(exc)
    return led, None


def _rescale_range(spec):
    lo, hi = spec.rescale_range
    if not (0 < lo < hi):
        raise SpecError("--rescale-range needs 0 < MIN < MAX")
    return lo, hi


def _resync(path, keys, spec, res):
    """The crop re-alignment search (wmcore/resync.py) over `keys`, if it is on: an ID or None. Runs only after the
    plain decodes found nothing, and before the (far slower) resize search."""
    if not spec.resync or not keys:
        return None
    res.resync_tried = True          # NOTE: with --resync on the plain DCT decodes of MIRRORED copies are not run at all
                                     # (see the callers): this search covers every mirroring, for every key, in one pass
    img = Image.open(path).convert("RGB")
    match = wmcore.find_spread_spectrum_id_resync(img, keys, max_seconds=spec.resync_budget_seconds)
    if match.text is None:
        res.resync_budget_exhausted = match.budget_exhausted
        return None
    res.regrid = match.shift if match.cropped else None       # a read the plain decoder could have made is not "cropped"
    res.flipped = None if match.orientation == "none" else match.orientation
    return match.text


# Share of --rescale-budget-seconds the scale search of wmcore/resync.py may use. It runs FIRST, before the established
# top-left-anchored resize search (about 0.5 s per ledger key when there is nothing to find, 1-3 s when there is):
# that search stops only on a hit or on the budget, so run first it would spend most of the budget on every copy it
# cannot read - every crop+resize copy - and on a long ledger even on plain resized ones whose key comes late.
_SCALE_SEARCH_SHARE = 0.4


def _resync_scaled(path, keys, spec, res):
    """The scale + crop re-alignment search (wmcore/resync.py), for --rescale with resync on: (ID or None,
    ran_out_of_time). It reads an ID from a resized copy whether or not it was also cropped, and reads nothing
    else (free-text payloads stay with the established search, which runs next when this finds nothing). The one
    thing the established search reports that this does not is its heuristic `confidence` figure."""
    if not spec.resync or not keys:
        return None, False
    res.resync_tried = True
    img = Image.open(path).convert("RGB")
    match = wmcore.find_spread_spectrum_id_resync_scaled(
        img, keys, scale_range=_rescale_range(spec), max_seconds=spec.rescale_budget_seconds * _SCALE_SEARCH_SHARE)
    if match.text is None:
        if match.budget_exhausted:
            res.scaled_keys_reached = match.keys_reached
        return None, match.budget_exhausted
    res.rescaled_by = match.factor if abs(match.factor - 1.0) > 1e-9 else None
    res.regrid = match.shift if match.cropped else None
    return match.text, False


def _spread_with_given_key(path, spec, res):
    """The single-key decode --ss-key has always meant (plain, or with the resize search), now also
    undoing a mirroring (left-right / top-bottom / both) - IDs only for the mirrored orientations, see
    wmcore/orient.py. With --rescale, the mirrored orientations are checked first (they cost a few block DCTs,
    against the search's seconds) and the resize search then runs exactly as before on the image as given:
    a copy that was mirrored AND resized is not searched (measured, bench/results/P3-NOTES.md)."""
    if spec.rescale:
        if not spec.resync:
            mirrored = extract_spread_spectrum_oriented(path, spec.ss_key, max_length=spec.max_length,
                                                        orientations=("h", "v", "hv"))
            if mirrored.text is not None:
                res.flipped = mirrored.orientation
                return mirrored.text
        cropped = _resync(path, [spec.ss_key], spec, res)
        if cropped is not None:
            return cropped
        t0 = time.monotonic()
        scaled, scaled_out_of_time = _resync_scaled(path, [spec.ss_key], spec, res)
        if scaled is not None:
            res.rescale_tried = True
            return scaled
        match = extract_spread_spectrum_rescaled(
            path, spec.ss_key, max_length=spec.max_length, scale_range=_rescale_range(spec),
            max_seconds=max(0.0, spec.rescale_budget_seconds - (time.monotonic() - t0)),
        )
        res.budget_exhausted, res.rescale_tried = match.budget_exhausted or scaled_out_of_time, True
        if match.text is not None:
            res.confidence = match.confidence
            if abs(match.factor - 1.0) > 1e-9:
                res.rescaled_by = match.factor
        return match.text
    found = extract_spread_spectrum_oriented(path, spec.ss_key, max_length=spec.max_length,
                                             orientations=("none",) if spec.resync else ORIENTATIONS)
    if found.text is not None and found.orientation != "none":
        res.flipped = found.orientation
    if found.text is None:
        return _resync(path, [spec.ss_key], spec, res)
    return found.text


def _spread_with_ledger_keys(path, keys, spec, res):
    """Try every key the ledger recorded, newest first, accepting only a well-formed watermark ID (a
    wrong key can decode junk - or, roughly one time in four (a bit more for an ID), the right payload; the ID's check
    characters are what tells them apart). Plain decodes share one DCT of the image, so this is cheap
    even for a long ledger; with --rescale the resize search then runs key by key against ONE shared
    time budget, so a long ledger can end in 'inconclusive' rather than run for hours."""
    res.keys_available = len(keys)
    # With --resync on, only the unmirrored plain pass runs here (one DCT, and the only pass that can read a free-text
    # payload). Mirrored copies are left to the resync search below, which covers every mirroring for every key in one
    # pass AND picks the key and orientation by confidence - the plain mirrored passes took the first key that read
    # anything, which for a wrong key reading the real ID (common, see wmcore/resync.py) labelled the orientation at
    # random. Measured: bench/results/CROP-NOTES.md §5b.
    for _key, text, orientation in iter_spread_spectrum_payloads_oriented(
            path, keys, max_length=spec.max_length, orientations=("none",) if spec.resync else ORIENTATIONS):
        if orientation == "none":               # keys_tried counts keys, not (key, mirroring) attempts
            res.keys_tried += 1
        if text is not None and wmid.parse(text):
            res.flipped = None if orientation == "none" else orientation
            return text
    cropped = _resync(path, keys, spec, res)
    if cropped is not None:
        return cropped
    if not spec.rescale:
        return None
    scale_range = _rescale_range(spec)
    res.rescale_tried = True
    res.keys_tried = 0
    t0 = time.monotonic()
    scaled, scaled_out_of_time = _resync_scaled(path, keys, spec, res)
    if scaled is not None:
        return scaled
    deadline = t0 + spec.rescale_budget_seconds
    for key in keys:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            res.budget_exhausted = True
            break
        # IDs only: this loop accepts nothing else, so a junk text hit must not be able to end a key's search early.
        match = extract_spread_spectrum_rescaled(path, key, max_length=spec.max_length, scale_range=scale_range,
                                                 max_seconds=remaining, decoder=wmcore.find_spread_spectrum_id)
        res.keys_tried += 1
        if match.text is not None and wmid.parse(match.text):
            res.confidence = match.confidence
            if abs(match.factor - 1.0) > 1e-9:
                res.rescaled_by = match.factor
            return match.text
        if match.budget_exhausted:
            res.budget_exhausted = True
            break
    res.budget_exhausted = res.budget_exhausted or scaled_out_of_time
    return None


def _id_from_c2pa(path):
    if not c2pa_provenance.AVAILABLE:
        return None
    try:
        return wmid.parse(c2pa_provenance.read_manifest(path).get("wm_id"))
    except Exception:
        return None                 # not a signable format, corrupt manifest...: no ID from this layer


def _c2pa_layer(path):
    if not c2pa_provenance.AVAILABLE:
        return _skipped("c2pa", "C2PA support isn't installed: pip install c2pa-python cryptography")
    return _read("c2pa", _id_from_c2pa(path))


def _spread_layer(text, res, *, from_ledger, resize_skipped):
    """The spread-spectrum layer's own outcome. Must be called BEFORE the headline logic clears res.flipped /
    res.rescaled_by / res.confidence for a result that lost to another layer: those say how THIS layer's
    mark was recovered, whichever layer the headline ends up quoting."""
    bits = []
    if res.flipped:
        bits.append(f"recovered after flipping the image back: this copy had been {ORIENTATION_LABELS[res.flipped]}")
    if res.regrid is not None:
        bits.append("recovered after re-aligning to the original 8x8 block grid: this copy looks cropped")
    if res.rescaled_by is not None:
        bits.append(f"recovered after resampling the image by x{res.rescaled_by:.4f}")
    if from_ledger and res.keys_available:
        bits.append(f"{res.keys_tried} of {res.keys_available} ledger key(s) tried")
    if resize_skipped:
        bits.append("resize search skipped: the LSB layer read the ID, and a resize destroys an LSB mark")
    note = "; ".join(bits) or None
    if text is None and (res.budget_exhausted or res.resync_budget_exhausted):
        return LayerResult("spread", "inconclusive", note=note)
    return _read("spread", text, note)


def _no_spread_reason(spec, keys):
    return ("--mode lsb looks only at the LSB mark" if spec.mode == "lsb"
            else "needs a key: pass --ss-key, or use a ledger that has recorded spread-spectrum keys")


def _extract_video(path, spec, ledger, ledger_error):
    """extract()'s video branch: video only ever carries the ID in the invisible spread-spectrum
    mark (video.scan_invisible - every frame, not just the first hit, see its docstring) or the C2PA
    manifest - there is no LSB layer (never survives re-encoding) and no file-format metadata layer
    for video the way EXIF/PNG text give images one."""
    keys = [spec.ss_key] if spec.ss_key is not None else (ledger.spread_keys() if ledger else [])
    if spec.mode == "spread" and not keys:
        raise SpecError("--mode spread requires --ss-key (or a ledger that has recorded spread-spectrum keys)")

    res = ExtractResult(found=False, ss_key_supplied=spec.ss_key is not None,
                        ledger_path=ledger.path if ledger else None, ledger_error=ledger_error,
                        show_layers=spec.layers)
    res.keys_available = len(keys)
    best_scan, text = None, None
    if spec.mode in ("spread", "auto") and keys:
        for key in keys:
            scan = video_mod.scan_invisible(path, key, jitter_refresh_seconds=spec.jitter_refresh_seconds,
                                            max_length=spec.max_length, resync=spec.resync)
            res.keys_tried += 1
            if best_scan is None:
                best_scan = scan            # keep the FIRST key's frame counts even if it's the one that misses,
                                             # so a "not found" report still says how much was actually checked
            if scan.wm_id is not None:
                best_scan, text = scan, scan.wm_id
                break

    # Each layer on its own. Video has no LSB layer (never survives re-encoding) and no file-format metadata
    # layer; C2PA is checked in auto mode even when the frames already gave an ID, so a disagreement shows.
    if best_scan is not None:
        frames = f"{best_scan.hits}/{best_scan.frames_checked} frame(s) decoded this ID" if text else (
            f"0/{best_scan.frames_checked} frame(s) checked decoded an ID")
        spread_layer = _read("spread", text, frames)
    else:
        spread_layer = _skipped("spread", _no_spread_reason(spec, keys) if spec.mode != "lsb"
                                else "--mode lsb looks only at the LSB mark, which video does not have")
    c2pa_layer = (_c2pa_layer(path) if spec.mode == "auto"
                  else _skipped("c2pa", "only --mode auto looks at the C2PA manifest"))
    layers = [_skipped("lsb", "video has no LSB layer: it would not survive re-encoding"), spread_layer,
              _skipped("metadata", "video has no file-format metadata layer"), c2pa_layer]

    scheme = "spread" if text else None
    if spec.mode == "auto" and text is None and c2pa_layer.wm_id:
        text, scheme = c2pa_layer.wm_id, "c2pa"

    res.found, res.text, res.scheme, res.video_scan = text is not None, text, scheme, (
        best_scan if scheme in (None, "spread") else None  # a c2pa hit isn't a frame-scan result - don't report frame counts for it
    )
    res.layers = layers
    canon = wmid.parse(text) if text is not None else None
    if canon:
        res.wm_id, res.text = canon, canon
        if ledger is not None:
            res.record = ledger.lookup(canon)
            if res.record and res.record.get("output_sha256"):
                res.exact_match = ledger_mod.file_sha256(path) == res.record["output_sha256"]
    return res


def extract(path, spec=None, **options):
    """Look for a hidden payload in `path`.

        wmapp.extract("out.png")                                   # LSB
        wmapp.extract("out.jpg", mode="spread", ss_key=20260918)   # spread-spectrum only
        wmapp.extract("small.jpg", mode="spread", ss_key=20260918, rescale=True)  # ...and search for a resize
        wmapp.extract("suspect.jpg")                               # a file stamped with --id: ID + ledger record
        wmapp.extract("suspect.mp4", ss_key=20260918)              # video: scans every frame, see video_scan

    mode="auto" (the default) tries LSB first and only falls through to spread-spectrum if LSB
    found nothing - or found something that isn't a watermark ID, since the LSB decoder can return
    a short junk "payload" from an image that carries no LSB mark at all (see bench/README.md,
    "false positives") and an ID's check characters are how that junk is told from a real mark.
    The key is --ss-key if given, else each key in the ledger. Only in auto mode, if no pixel
    layer produced an ID, the file's metadata and C2PA manifest are checked for one too. Pass
    mode="spread" (or "lsb") when you know which kind of mark you embedded. For a video, LSB and
    metadata don't apply (see _extract_video) - only spread and c2pa are tried, and a "found" via
    the spread layer carries `video_scan`, the frame-by-frame breakdown (see VideoScanResult).

    When the payload is a watermark ID the result carries `wm_id`, the ledger `record` (if a ledger
    was consulted and knows the ID) and `exact_match` (is this file byte-identical to what was
    stamped). The ledger is $WATERMARK_TOOL_LEDGER / ~/.watermark-tool/ledger.sqlite3 if that
    exists, or `ledger=` explicitly; `use_ledger=False` skips it.

    Returns an ExtractResult (found/text/scheme/...); `not result.found` is a
    normal outcome, not an error. Raises SpecError only for a request that
    can't make sense (mode="spread" with neither a key nor a ledger that has any, a bad rescale range)."""
    if isinstance(spec, ExtractSpec):
        spec = spec.replace(**options) if options else spec
    else:
        data = dict(spec or {})
        data.update(options)
        spec = ExtractSpec.from_mapping(data)

    ledger, ledger_error = _open_ledger(spec)

    if os.path.splitext(path)[1].lower() in wmcore.VIDEO_EXTS:
        return _extract_video(path, spec, ledger, ledger_error)

    keys = [spec.ss_key] if spec.ss_key is not None else (ledger.spread_keys() if ledger else [])
    if spec.mode == "spread" and not keys:
        raise SpecError("--mode spread requires --ss-key (or a ledger that has recorded spread-spectrum keys)")

    res = ExtractResult(found=False, ss_key_supplied=spec.ss_key is not None,
                        ledger_path=ledger.path if ledger else None, ledger_error=ledger_error,
                        show_layers=spec.layers)
    text = scheme = None

    # Every layer is checked on its own, then the headline (text/scheme/...) is chosen from them exactly the way
    # it always was: LSB, unless it gave no ID and spread-spectrum did, unless neither did and metadata / C2PA did.
    if spec.mode in ("lsb", "auto"):
        text = extract_invisible(path, max_length=spec.max_length)
        scheme = "lsb" if text is not None else None
        lsb_layer = _read("lsb", text)
    else:
        lsb_layer = _skipped("lsb", "--mode spread looks only at the spread-spectrum mark")

    have_id = text is not None and wmid.parse(text) is not None
    if spec.mode in ("spread", "auto") and keys:
        # The one shortcut: the resize search (--rescale) is not run when the LSB layer already read a real ID. A resize
        # destroys an LSB mark (measured: 75% -> nothing readable, bench/results/EXTRACT-LAYERS-NOTES.md), so that
        # search cannot find anything on such a copy and would only spend --rescale-budget-seconds (20s by default) on
        # a "not found". The FLIP search still runs: a top-bottom flip does not destroy an LSB mark (measured).
        resize_skipped = bool(spec.rescale) and have_id
        run_spec = spec.replace(rescale=False) if resize_skipped else spec
        spread_text = (_spread_with_given_key(path, run_spec, res) if spec.ss_key is not None
                       else _spread_with_ledger_keys(path, keys, run_spec, res))
        spread_layer = _spread_layer(spread_text, res, from_ledger=spec.ss_key is None, resize_skipped=resize_skipped)
        # A spread result replaces an LSB one only if there was no LSB one, or the spread one is a real ID
        # (legacy behaviour: an LSB text is never overridden by a non-ID spread text) - and never an LSB ID.
        if spread_text is not None and not have_id and (text is None or wmid.parse(spread_text)):
            text, scheme = spread_text, "spread"
        else:
            res.confidence = res.rescaled_by = res.flipped = res.regrid = None
        have_id = text is not None and wmid.parse(text) is not None
    else:
        spread_layer = _skipped("spread", _no_spread_reason(spec, keys))

    if spec.mode == "auto":
        meta_layer = _read("metadata", wmid.parse(wmcore.metadata.read_wm_id_field(path)))
        c2pa_layer = _c2pa_layer(path)
        if not have_id:
            from_meta, from_c2pa = meta_layer.wm_id, c2pa_layer.wm_id
            if from_meta or from_c2pa:
                text, scheme = from_meta or from_c2pa, ("metadata" if from_meta else "c2pa")
                res.rescaled_by = res.confidence = res.flipped = res.regrid = None
    else:
        meta_layer = _skipped("metadata", "only --mode auto looks at metadata")
        c2pa_layer = _skipped("c2pa", "only --mode auto looks at the C2PA manifest")

    res.layers = [lsb_layer, spread_layer, meta_layer, c2pa_layer]
    res.found, res.text, res.scheme = text is not None, text, scheme
    canon = wmid.parse(text) if text is not None else None
    if canon:
        res.wm_id, res.text = canon, canon
        if ledger is not None:
            res.record = ledger.lookup(canon)
            if res.record and res.record.get("output_sha256"):
                res.exact_match = ledger_mod.file_sha256(path) == res.record["output_sha256"]
    return res


def verify(path):
    """Read the C2PA manifest of `path` -> the parsed summary dict
    (c2pa_provenance.read_manifest). Raises UnavailableError if C2PA support
    isn't installed."""
    if not c2pa_provenance.AVAILABLE:
        raise UnavailableError("C2PA support needs two extra packages: pip install c2pa-python cryptography")
    return c2pa_provenance.read_manifest(path)


def capabilities():
    """What this installation can do - the optional pieces. Served to the web
    page so it can hide controls that would fail."""
    return {
        "c2pa": bool(c2pa_provenance.AVAILABLE),
        "exif": wmcore.metadata.piexif is not None,
        "ffmpeg": shutil.which("ffmpeg") is not None,
    }
