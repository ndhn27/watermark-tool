"""wmapp.pipeline - the watermarking use-cases.

This is the application layer: it turns a validated request (WatermarkSpec)
into a result, and it is the ONLY place that knows how the pieces fit
together - grid layer, adaptive map, blend, invisible mark, metadata, C2PA,
video dispatch, batch fan-out. The CLI (watermark.py), the HTTP adapter
(serve.py) and Python callers are thin shells around it: they parse their own
input format into a WatermarkSpec, call one of these functions, and render
the result their own way. Before this module the pipeline lived inside
watermark.py's apply_watermark() and main(), mixed with argparse handling and
print() calls, which is why nothing else could reuse it.

Entry points:
    watermark(input, output, spec=None, **options)   one file (image or video)
    watermark_batch(in_dir, out_dir, spec, ...)      a folder
    prepare(spec, ...)                               resolve run-level things once
                                                     (dev cert, random ss_key)

prepare() exists because a few decisions must be made ONCE per run, not per
file: a random --ss-key shared by every file in a batch, the dev certificate
created before any worker process starts (creating it isn't lock-protected;
see c2pa_provenance.get_or_create_dev_cert()), and - with --id - the ledger
opened (so a bad path fails before any work). The watermark ID itself is NOT
a run-level decision: every output file gets its own, allocated in
watermark_file(). watermark() calls it for you;
watermark_batch() and long-lived callers call it explicitly and reuse the
Prepared result.
"""

import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

import c2pa_provenance
import video as video_mod
import wmcore

from wmcore import wmid

from .errors import SpecError, UnavailableError
from .events import NoteBus
from .ledger import Ledger
from .limits import NO_LIMITS
from .options import WatermarkSpec


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Prepared:
    """A spec whose run-level decisions have been made, plus the notes that
    making them produced. Immutable, so it can be shared across a batch."""
    spec: WatermarkSpec
    notes: tuple = ()
    batch: bool = False


@dataclass
class FileResult:
    input: str
    output: str
    kind: str                                   # "image" | "video"
    seconds: float = 0.0
    ss_key: Optional[int] = None                # the key needed to read a spread-spectrum mark back
    c2pa: str = "off"                           # "off" | "signed" | "failed"
    notes: list = field(default_factory=list)
    wm_id: Optional[str] = None                 # the ID stamped into this output (--id), canonical form
    id_layers: list = field(default_factory=list)   # which layers actually carry it: grid, lsb|spread, metadata, c2pa

    def to_dict(self):
        return {"input": self.input, "output": self.output, "kind": self.kind,
                "seconds": round(self.seconds, 3), "ss_key": self.ss_key, "c2pa": self.c2pa,
                "wm_id": self.wm_id, "id_layers": list(self.id_layers),
                "notes": [n.to_dict() for n in self.notes]}


@dataclass
class BatchResult:
    output_dir: str
    results: list = field(default_factory=list)      # FileResult per success
    failures: list = field(default_factory=list)     # (input_path, error string) per failure
    ss_key: Optional[int] = None
    notes: list = field(default_factory=list)

    @property
    def ok(self):
        return not self.failures

    def to_dict(self):
        return {"output_dir": self.output_dir, "ss_key": self.ss_key, "ok": self.ok,
                "results": [r.to_dict() for r in self.results],
                "failures": [{"input": p, "error": e} for p, e in self.failures],
                "notes": [n.to_dict() for n in self.notes]}


# ---------------------------------------------------------------------------
# prepare(): decisions made once per run
# ---------------------------------------------------------------------------

def _as_spec(spec, options):
    if isinstance(spec, WatermarkSpec):
        return spec.replace(**options) if options else spec
    data = dict(spec or {})
    data.update(options)
    return WatermarkSpec.from_mapping(data)


def prepare(spec=None, *, batch=False, on_note=None, **options):
    """Validate `spec` (a WatermarkSpec, a dict, or keyword options) and resolve
    everything that must be decided once per run. Raises SpecError for a
    contradictory request and UnavailableError if --c2pa was asked for but
    its packages aren't installed."""
    if isinstance(spec, Prepared):
        return spec
    spec = _as_spec(spec, options)
    bus = NoteBus(on_note)

    if batch and spec.adaptive_preview:
        bus.info("adaptive_preview_batch",
                 "Note: --adaptive-preview isn't supported with --batch (one path, many images); ignored.")
        spec = spec.replace(adaptive_preview=None)

    if bool(spec.c2pa_cert) != bool(spec.c2pa_key):
        raise SpecError("--c2pa-cert and --c2pa-key must be given together")
    if (spec.c2pa_cert or spec.c2pa_key or spec.c2pa_title
            or spec.c2pa_alg != c2pa_provenance.DEFAULT_ALG) and not spec.c2pa:
        bus.info("c2pa_options_ignored",
                 "Note: --c2pa-cert/--c2pa-key/--c2pa-alg/--c2pa-title have no effect without --c2pa; ignored.")

    if spec.c2pa and not c2pa_provenance.AVAILABLE:
        raise UnavailableError(
            "--c2pa requires two extra packages: pip install c2pa-python cryptography "
            "(both are listed, commented out, in requirements.txt)"
        )

    if spec.c2pa and not spec.c2pa_cert:
        # Resolve (and generate, if this is the very first run) the dev cert
        # HERE - once, before any batch worker pool spawns - rather than
        # leaving it to sign_file()'s own lazy fallback: the check-then-write
        # that creates it isn't lock-protected, so several worker processes
        # racing to generate the same missing files could corrupt one or both.
        cert_path, key_path, created = c2pa_provenance.get_or_create_dev_cert()
        spec = spec.replace(c2pa_cert=cert_path, c2pa_key=key_path)
        if spec.c2pa_alg != c2pa_provenance.DEFAULT_ALG:
            bus.info(
                "c2pa_alg_ignored",
                f"Note: --c2pa-alg {spec.c2pa_alg} ignored - the auto-generated dev "
                f"certificate (no --c2pa-cert given) is {c2pa_provenance.DEFAULT_ALG}-only. "
                f"Pass --c2pa-cert/--c2pa-key with a matching certificate to sign with a "
                f"different algorithm.",
            )
            spec = spec.replace(c2pa_alg=c2pa_provenance.DEFAULT_ALG)
        bus.info(
            "c2pa_dev_cert",
            f"Note: --c2pa is signing with a local, self-signed dev certificate "
            f"({'generated just now' if created else 'reused'} at {cert_path}). The "
            f"signature itself is real and tamper-evident, but C2PA-aware readers "
            f"(including cli/c2pa_verify.py) will report the signing credential as "
            f"untrusted rather than showing a verified checkmark, until you pass "
            f"--c2pa-cert/--c2pa-key for a certificate issued by a CA in the C2PA trust "
            f"list. See README.md's C2PA section.",
            cert_path=cert_path, created=created,
        )

    if spec.stamp_id:
        spec = _prepare_id(spec, batch, bus)
    elif spec.id_value is not None or spec.grid_id or spec.ledger:
        bus.info("id_options_ignored",
                 "Note: --id-value/--grid-id/--ledger have no effect without --id; ignored.")

    if spec.invisible and spec.invisible_mode == "spread" and spec.ss_key is None:
        key = random.SystemRandom().randrange(1, 2**31)
        spec = spec.replace(ss_key=key)
        bus.info(
            "ss_key_generated",
            f"Note: no --ss-key given, so a random one was generated for this run: {key}. "
            f"SAVE THIS - it's required to read the spread-spectrum mark back later, with "
            f"extract_invisible.py --mode spread --ss-key {key}. Every file in this run "
            f"(including a --batch run) shares this one key.",
            ss_key=key,
        )

    return Prepared(spec, tuple(bus.items), batch)


def _prepare_id(spec, batch, bus):
    """The --id half of prepare(): validate an explicit ID, open (creating if need be) the ledger so a
    bad path fails before any work, and say where IDs are being recorded."""
    if spec.id_value is not None:
        if batch:
            raise SpecError("id_value: one ID can't be given to a whole --batch; every file needs its own "
                            "(leave --id-value off and each gets a fresh one)")
        canon = wmid.parse(spec.id_value)
        if canon is None:
            raise SpecError(f"id_value: {spec.id_value!r} is not a valid watermark ID "
                            f"(10 characters from {wmid.ALPHABET}, including its 2 check characters)")
        spec = spec.replace(id_value=canon)
    ledger = Ledger(spec.ledger)
    spec = spec.replace(ledger=ledger.path)
    bus.info("id_ledger",
             f"Note: --id records every output in the ledger at {ledger.path}. Keep that file: it is what turns "
             f"an ID found in a suspect copy back into a file, a date and (for --invisible-mode spread) the "
             f"key. It holds those keys, so don't share it.",
             ledger=ledger.path)
    if spec.invisible and spec.invisible_text:
        bus.info("id_text_ignored",
                 "Note: --invisible-text is ignored with --id: the hidden mark carries the ID, and what "
                 "the text would have said belongs in the ledger's record instead.")
    if not spec.invisible:
        bus.info("id_no_hidden_mark",
                 "Note: --id without --invisible puts the ID in image metadata, the C2PA manifest (--c2pa) and "
                 "the visible grid (--grid-id) only - all of which are easy to lose. Add --invisible "
                 "--invisible-mode spread for the layer that survives recompression.")
    return spec


# ---------------------------------------------------------------------------
# One file
# ---------------------------------------------------------------------------

def _watermark_image(input_path, output_path, spec, bus, limits):
    """The still-image pipeline (formerly watermark.apply_watermark)."""
    limits.check_text(spec.text)
    base = Image.open(input_path)
    limits.check_image(base.size, spec)          # header only - nothing decoded yet
    base = base.convert("RGBA")

    sensitivity_map = None
    if spec.adaptive:
        rgb = np.asarray(base.convert("RGB"))
        sensitivity_map = wmcore.compute_sensitivity_map(
            rgb, face_cascade_path=spec.face_cascade, detail_weight=spec.adaptive_detail_weight,
        )
        if spec.adaptive_preview:
            Image.fromarray((sensitivity_map * 255).astype(np.uint8), "L").save(spec.adaptive_preview)

    wm_id = spec.id_value if spec.stamp_id else None
    grid_text = _grid_text(spec, wm_id)
    layers = ["grid"] if (wm_id and spec.grid_id) else []

    watermark_layer = wmcore.create_watermark_layer(
        base.size, grid_text, spec.font_size, spec.opacity, spec.angle, spec.spacing, spec.font_path,
        seed=spec.seed, sensitivity_map=sensitivity_map, adaptive_strength=spec.adaptive_strength,
        infill_spacing_divisor=spec.adaptive_infill_divisor,
    )
    ink_for_blend = wmcore.build_ink_map(spec.ink, sensitivity_map, spec.adaptive_strength, spec.blend)
    watermarked = wmcore.composite(base, watermark_layer, mode=spec.blend, ink=ink_for_blend)

    ss_key = None
    if spec.invisible:
        payload_text = wm_id or spec.invisible_text or spec.text
        layers.append(spec.invisible_mode)
        if spec.invisible_mode == "spread":
            if spec.ss_key is None:
                raise SpecError(
                    "invisible_mode='spread' requires ss_key - prepare() resolves this once per "
                    "run (see --ss-key) rather than each call picking its own."
                )
            watermarked = wmcore.embed_spread_spectrum(
                watermarked, payload_text, spec.ss_key,
                alpha=spec.ss_alpha if spec.ss_alpha is not None else wmcore.DEFAULT_SS_ALPHA,
            )
            ss_key = spec.ss_key
        else:
            if output_path.lower().endswith((".jpg", ".jpeg")):
                bus.warning(
                    "lsb_lossy_format",
                    f"Warning: --invisible-mode lsb requires a lossless format; "
                    f"{output_path} is JPEG and recompression will destroy it. "
                    f"Consider a .png output, or --invisible-mode spread which "
                    f"survives JPEG recompression.",
                )
            watermarked = wmcore.embed_invisible(watermarked, payload_text)

    wmcore.save_with_metadata(
        watermarked, output_path, author=spec.author, copyright_text=spec.copyright_text,
        notify=lambda message: bus.warning("exif_unavailable", message), wm_id=wm_id,
    )
    if wm_id and wmcore.metadata.writes_wm_id(output_path):
        layers.append("metadata")

    c2pa_status = "off"
    if spec.c2pa:
        ok = c2pa_provenance.sign_file_or_note(
            output_path, title=spec.c2pa_title, author=spec.author, parent_path=input_path,
            cert_path=spec.c2pa_cert, key_path=spec.c2pa_key, alg=spec.c2pa_alg,
            notify=lambda message: bus.warning("c2pa_failed", message), wm_id=wm_id,
        )
        c2pa_status = "signed" if ok else "failed"
        if ok and wm_id:
            layers.append("c2pa")
    return ss_key, c2pa_status, layers


def _grid_text(spec, wm_id):
    """The visible grid's text: --text, plus the ID in its display form with --grid-id."""
    if wm_id and spec.grid_id:
        return f"{spec.text} {wmid.display(wm_id)}"
    return spec.text


def _watermark_video(input_path, output_path, spec, bus, workers, show_progress):
    """Video dispatch (formerly watermark.apply_watermark_any's video branch)."""
    ignored = []
    if spec.copyright_text:
        ignored.append("--copyright")
    # --author is the one option in this list that ISN'T simply dropped for
    # video when --c2pa is on: nothing writes it into file-format metadata (no
    # video equivalent of EXIF / PNG text), but it does reach the C2PA
    # CreativeWork assertion. So it's only truly ignored when --c2pa is off.
    if spec.author and not spec.c2pa:
        ignored.append("--author")
    if ignored:
        plural = "are" if len(ignored) > 1 else "is"
        bus.info("video_options_ignored", f"Note: {'/'.join(ignored)} {plural} image-only; ignored for {input_path}.")
    if spec.adaptive_preview:
        bus.info("adaptive_preview_video", f"Note: --adaptive-preview is images-only; ignored for {input_path}.")

    wm_id = spec.id_value if spec.stamp_id else None
    layers = ["grid"] if (wm_id and spec.grid_id) else []

    ss_key = ss_alpha = ss_payload = None
    if spec.invisible:
        if spec.invisible_mode == "lsb":
            bus.warning("video_lsb_unsupported",
                        f"Warning: --invisible-mode lsb doesn't survive video re-encoding (H.264 or the "
                        f"OpenCV mp4v fallback are both lossy) - no hidden mark was embedded for "
                        f"{output_path}. Use --invisible-mode spread for video.")
        else:
            if spec.ss_key is None:
                raise SpecError(
                    "invisible_mode='spread' requires ss_key - prepare() resolves this once per "
                    "run (see --ss-key) rather than each call picking its own."
                )
            ss_key = spec.ss_key
            ss_alpha = spec.ss_alpha if spec.ss_alpha is not None else wmcore.DEFAULT_SS_ALPHA_VIDEO
            ss_payload = wm_id or spec.invisible_text or spec.text
            layers.append("spread")

    if wm_id and not spec.grid_id and not spec.c2pa and ss_key is None:
        bus.warning("video_id_not_embedded",
                    f"Warning: --id on a video only puts the ID in the visible grid (--grid-id) and the C2PA "
                    f"manifest (--c2pa) - neither was asked for, and no hidden pixel mark was embedded either "
                    f"(--invisible --invisible-mode spread), so the ID recorded for {output_path} is written "
                    f"into nothing the file carries.")

    video_mod.apply_watermark_video(
        input_path, output_path, workers=workers, show_progress=show_progress,
        text=_grid_text(spec, wm_id), opacity=spec.opacity, font_size=spec.font_size, angle=spec.angle,
        spacing=spec.spacing, font_path=spec.font_path, seed=spec.seed, blend=spec.blend, ink=spec.ink,
        use_shared_memory=spec.use_shared_memory, jitter_refresh_seconds=spec.jitter_refresh_seconds,
        adaptive=spec.adaptive, adaptive_strength=spec.adaptive_strength,
        adaptive_detail_weight=spec.adaptive_detail_weight, face_cascade=spec.face_cascade,
        adaptive_infill_divisor=spec.adaptive_infill_divisor,
        adaptive_track=spec.adaptive_track, adaptive_redetect_seconds=spec.adaptive_redetect_seconds,
        scene_cut_detect=spec.scene_cut_detect, scene_cut_threshold=spec.scene_cut_threshold,
        c2pa=spec.c2pa, c2pa_cert=spec.c2pa_cert, c2pa_key=spec.c2pa_key,
        c2pa_alg=spec.c2pa_alg, c2pa_title=spec.c2pa_title,
        author=spec.author if spec.c2pa else None, wm_id=wm_id,
        ss_key=ss_key, ss_alpha=ss_alpha, ss_payload=ss_payload,
        notify=lambda message, level="info", code=None: bus.add(level, code or "video", message),
    )
    # video.py reports signing problems as a "c2pa_failed" note.
    c2pa_status = ("off" if not spec.c2pa else
                   "failed" if any(n.code == "c2pa_failed" for n in bus.items) else "signed")
    if wm_id and c2pa_status == "signed":
        layers.append("c2pa")
    return ss_key, c2pa_status, layers


def watermark_file(input_path, output_path, prepared, *, workers=None, show_progress=False,
                   limits=NO_LIMITS, on_note=None):
    """Watermark one file - an image or (by extension) a video - with an
    already-prepared spec. Returns a FileResult."""
    spec = prepared.spec
    bus = NoteBus(on_note)
    kind = "video" if os.path.splitext(input_path)[1].lower() in wmcore.VIDEO_EXTS else "image"
    t0 = time.perf_counter()

    ledger = wm_id = None
    if spec.stamp_id:
        # Reserve the ID (a 'pending' ledger row) BEFORE anything is stamped, so no file can leave with
        # an ID the ledger has never heard of; complete it with the output's hash after every layer is applied.
        ledger = Ledger(spec.ledger)
        hidden = spec.invisible and (kind == "image" or spec.invisible_mode == "spread")
        spread = hidden and spec.invisible_mode == "spread"
        resolved_alpha = None
        if spread:
            default_alpha = wmcore.DEFAULT_SS_ALPHA if kind == "image" else wmcore.DEFAULT_SS_ALPHA_VIDEO
            resolved_alpha = spec.ss_alpha if spec.ss_alpha is not None else default_alpha
        wm_id = ledger.reserve(
            kind=kind, input_path=input_path, spec_dict=spec.to_dict(),
            invisible_mode=spec.invisible_mode if hidden else None,
            ss_key=spec.ss_key if spread else None, ss_alpha=resolved_alpha,
            wm_id=spec.id_value,
        )
        spec = spec.replace(id_value=wm_id)

    try:
        if kind == "video":
            ss_key, c2pa_status, layers = _watermark_video(input_path, output_path, spec, bus,
                                                           workers or os.cpu_count() or 1, show_progress)
        else:
            ss_key, c2pa_status, layers = _watermark_image(input_path, output_path, spec, bus, limits)
        if ledger is not None:
            ledger.complete(wm_id, output_path, layers)
    except BaseException as exc:
        if ledger is not None:
            try:
                ledger.fail(wm_id, f"{type(exc).__name__}: {exc}")
            except Exception:
                pass                        # the original failure is the one worth reporting
        raise

    if wm_id is None:
        layers = []                         # `layers` lists ID carriers; without an ID there are none
    else:
        carried = ", ".join(layers) if layers else "no layer of this file"
        readback = (f"python3 cli/extract_invisible.py {output_path}" if (kind == "image" or "spread" in layers)
                    else f"python3 cli/wm_ledger.py show {wm_id}")
        bus.info("id_stamped",
                 f"Watermark ID: {wmid.display(wm_id)} (carried by: {carried}); recorded in {ledger.path}. "
                 f"{'Read it back' if kind == 'image' else 'Look it up'} with: {readback}",
                 wm_id=wm_id, layers=list(layers), ledger=ledger.path)
    return FileResult(input_path, output_path, kind, time.perf_counter() - t0, ss_key, c2pa_status, list(bus.items),
                      wm_id=wm_id, id_layers=list(layers))


def watermark(input_path, output_path, spec=None, *, workers=None, show_progress=False,
              limits=NO_LIMITS, on_note=None, **options):
    """Watermark one file. The friendly entry point.

        import wmapp
        result = wmapp.watermark("in.jpg", "out.png", text="@me", blend="multiply",
                                 invisible=True, invisible_mode="spread")
        result.ss_key       # generated for you, since none was given
        result.notes        # everything the CLI would have printed

    `spec` may be a WatermarkSpec, a dict, or a Prepared (from prepare()); any
    extra keyword arguments are options and override the spec's. Nothing is
    printed unless you pass `on_note`."""
    bus = NoteBus(on_note)
    if isinstance(spec, Prepared):
        if options:
            raise SpecError("options can't be combined with an already-prepared spec; pass them to prepare()")
        prepared = spec
    else:
        prepared = prepare(spec, on_note=lambda note: bus.forward([note]), **options)
    result = watermark_file(input_path, output_path, prepared, workers=workers, show_progress=show_progress,
                            limits=limits, on_note=on_note)
    result.notes = list(bus.items) + result.notes
    return result


# ---------------------------------------------------------------------------
# A folder
# ---------------------------------------------------------------------------

def _stable_file_seed(base_seed, fname):
    """Deterministic per-file seed offset for batches, so every file gets its
    own jitter pattern instead of an identical (and itself detectable) one,
    while still being reproducible run-to-run for a given seed. Thin wrapper
    around wmcore.derive_seed() - see there for why crc32."""
    return wmcore.derive_seed(base_seed, fname)


def _batch_worker(job):
    """Worker for the batch ProcessPoolExecutor - one image per call. At module
    level (not a closure) so it pickles to worker processes; combined with the
    __main__ guard in the CLI scripts, the standard safe pattern for
    multiprocessing in a directly-run script (see wmcore's docstring)."""
    in_path, out_path, spec_dict, limits = job
    try:
        spec = WatermarkSpec.from_mapping(spec_dict)
        return in_path, watermark_file(in_path, out_path, Prepared(spec), limits=limits), None
    except Exception as exc:  # report failure, don't crash the whole batch
        return in_path, None, str(exc)


def watermark_batch(input_dir, output_dir, prepared, *, workers=None, show_progress=False,
                    limits=NO_LIMITS, on_note=None, on_progress=None):
    """Watermark every image and video in `input_dir` into `output_dir`.
    Images run in parallel across `workers` processes; videos one at a time
    (each already parallelises across frames). `on_progress(done, total)` is
    called once with (0, total) before the images start and after each one."""
    bus = NoteBus(on_note)
    if not isinstance(prepared, Prepared):
        prepared = prepare(prepared, batch=True, on_note=lambda note: bus.forward([note]))
    spec = prepared.spec
    workers = workers or os.cpu_count() or 1
    result = BatchResult(output_dir, ss_key=spec.ss_key if (spec.invisible and spec.invisible_mode == "spread") else None)

    os.makedirs(output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(wmcore.IMAGE_EXTS + wmcore.VIDEO_EXTS))
    if not files:
        bus.info("no_files", "No images or videos found in the input folder.")
        result.notes = list(bus.items)
        return result
    image_files = [f for f in files if f.lower().endswith(wmcore.IMAGE_EXTS)]
    video_files = [f for f in files if f.lower().endswith(wmcore.VIDEO_EXTS)]

    def per_file(fname):
        return spec.replace(seed=_stable_file_seed(spec.seed, fname)) if spec.seed is not None else spec

    if image_files:
        jobs = [(os.path.join(input_dir, f), os.path.join(output_dir, f), per_file(f).to_dict(), limits)
                for f in image_files]
        bus.info("batch_images_start", f"Watermarking {len(jobs)} image(s) across {workers} worker(s)...")
        if on_progress:
            on_progress(0, len(jobs))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for done, (in_path, res, err) in enumerate(pool.map(_batch_worker, jobs), start=1):
                if res is None:
                    result.failures.append((in_path, err))
                else:
                    result.results.append(res)
                    bus.forward(res.notes)
                if on_progress:
                    on_progress(done, len(jobs))
        for in_path, err in result.failures:
            bus.info("batch_file_failed", f"  FAILED: {in_path}: {err}", input=in_path)
        bus.info("batch_images_done", f"Done: {len(jobs) - len(result.failures)}/{len(jobs)} image(s) saved to {output_dir}")

    for fname in video_files:
        in_path, out_path = os.path.join(input_dir, fname), os.path.join(output_dir, fname)
        bus.info("batch_video_start", f"Watermarking video: {fname} (parallel across {workers} worker(s) for frames)")
        res = watermark_file(in_path, out_path, Prepared(per_file(fname)), workers=workers,
                             show_progress=show_progress, limits=limits, on_note=on_note)
        result.results.append(res)
        # A video used to announce itself ("Saved: ...") from inside video.py; batch
        # keeps that, single-file callers get the path from the result instead.
        bus.info("saved", f"Saved: {out_path}", output=out_path)

    result.notes = list(bus.items)
    return result
