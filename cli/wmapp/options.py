"""wmapp.options - the single source of truth for every user-facing option.

Before this module the same knobs were defined three times: as argparse flags
in watermark.py (with their own defaults and help), as keyword arguments of
apply_watermark()/apply_watermark_video() (with their own, separately typed
defaults), and - for opacity, spacing and the rest - as sliders in
web/index.html (with different units: the CLI's opacity is 0-255, the page's
is a percentage). Nothing forced them to agree.

Now there is one list, OPTIONS. From it are derived:

  * WatermarkSpec / ExtractSpec  the typed request objects the Python API takes
                                 (validated on construction, JSON round-trippable)
  * add_arguments()              the argparse flags of watermark.py / extract_invisible.py
  * schema()                     the JSON the HTTP adapter serves at /api/schema, from
                                 which the web page renders and range-checks its controls

so a new option, a changed default or a tightened range is made once, here.

Options carry a few flags beyond type/default/help:

  role="run"   orchestration (--batch, --workers): CLI/run-level, not part of a
               watermark request, so not in WatermarkSpec or the JSON API.
  requires     ((name, value), ...): the option only matters when those other options
               have those values (ss_key only with invisible=True and invisible_mode
               "spread"). Purely a hint for UIs - the pipeline ignores an irrelevant
               option - but it lets the web page show/hide controls from the schema.
  local=True   the value names a path on the machine running the code (font,
               certificate, preview output). Fine for the CLI and the Python
               API; the HTTP adapter refuses it, because a network client must
               not be able to make the server read or write arbitrary paths.
  min / max    hard bounds, deliberately generous (they exist so garbage and
               denial-of-service values are rejected uniformly, not to
               second-guess reasonable use). Derived from the ranges the old
               --help text already documented.
"""

import math
from dataclasses import dataclass, field, fields, make_dataclass, replace as _dc_replace
from typing import Any, Optional

import c2pa_provenance
import wmcore
from wmcore.invisible import MAX_PLAUSIBLE_LENGTH

from .errors import SpecError


@dataclass(frozen=True)
class Option:
    name: str
    type: str                       # "str" | "int" | "float" | "bool" | "choice" | "float_pair"
    default: Any = None
    help: str = ""
    flag: Optional[str] = None      # CLI flag when it isn't "--" + name with dashes
    negate: bool = False            # bool whose CLI flag stores False (--no-shared-memory)
    choices: Optional[tuple] = None
    min: Optional[float] = None
    max: Optional[float] = None
    group: str = "grid"
    scope: tuple = ("image", "video")
    local: bool = False
    required: bool = False
    role: str = "spec"              # "spec" | "run"
    metavar: Optional[tuple] = None
    requires: tuple = ()            # ((option, value), ...) - only meaningful when all hold (UI hint)

    @property
    def cli_flag(self):
        return self.flag or "--" + self.name.replace("_", "-")

    def to_dict(self):
        d = {
            "name": self.name, "type": self.type, "flag": self.cli_flag, "help": self.help,
            "default": list(self.default) if isinstance(self.default, tuple) else self.default,
            "group": self.group, "scope": list(self.scope), "local": self.local, "required": self.required,
        }
        if self.choices:
            d["choices"] = list(self.choices)
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        if self.requires:
            d["requires"] = {k: v for k, v in self.requires}
        return d


GROUPS = {
    "grid": "Visible grid",
    "blend": "Blend",
    "invisible": "Invisible mark",
    "id": "Watermark ID & ledger",
    "metadata": "Metadata",
    "adaptive": "Adaptive placement",
    "c2pa": "C2PA / Content Credentials",
    "video": "Video",
    "run": "Run",
}

# ---------------------------------------------------------------------------
# The options. Order matters: it is the order of the CLI's --help.
# Help strings are the CLI's, verbatim (a % here is a plain %; add_arguments()
# escapes it for argparse).
# ---------------------------------------------------------------------------

OPTIONS = (
    Option(
        "text", "str", None,
        help="Watermark text, e.g. @yourhandle or a group/brand name",
        required=True, group="grid",
    ),
    Option(
        "opacity", "int", 90,
        help="Base opacity, 0-255 (default: 90, ~35%)",
        min=0, max=255, group="grid",
    ),
    Option(
        "font_size", "int", 36,
        help="Font size (default: 36)",
        min=1, max=5000, group="grid",
    ),
    Option(
        "angle", "int", 30,
        help="Base rotation angle in degrees (default: 30)",
        min=-3600, max=3600, group="grid",
    ),
    Option(
        "spacing", "int", 250,
        help="Spacing between watermark tiles, in px (default: 250)",
        min=1, max=100000, group="grid",
    ),
    Option(
        "font_path", "str", None,
        help="Optional path to a custom .ttf font file",
        local=True, group="grid",
    ),
    Option(
        "batch", "bool", False,
        help="Process a whole input folder into an output folder",
        role="run", group="run",
    ),
    Option(
        "workers", "int", None,
        help=(
            "Parallel workers: files-in-parallel for --batch images, frames-in-parallel "
            "for a video (default: CPU count)"
        ),
        min=1, role="run", group="run",
    ),
    Option(
        "seed", "int", None,
        help="Random seed for tile jitter (reuse it to reproduce the same pattern)",
        group="grid",
    ),
    Option(
        "use_shared_memory", "bool", True,
        help=(
            "Video only: disable the shared-memory frame transport and fall back to the "
            "slower pickle-per-frame path (useful if shared memory isn't available in "
            "your environment)"
        ),
        flag="--no-shared-memory", negate=True, group="video", scope=("video",),
    ),
    Option(
        "jitter_refresh_seconds", "float", 4.0,
        help=(
            "Video only: regenerate the watermark's jitter pattern (new random tile "
            "positions/opacity/rotation) every N seconds of output instead of using one "
            "static pattern for the whole clip (default: 4.0). A single unchanging "
            "pattern stamped over moving footage is itself averageable out via a "
            "per-pixel median across enough frames, the same collusion/averaging attack "
            "the README describes for multiple images sharing a seed - just within one "
            "file instead of across several. Pass 0 to disable and use one static pattern "
            "for the whole video (old behavior; only meaningful with --seed set, for "
            "exact reproducibility)."
        ),
        min=0, group="video", scope=("video",),
    ),
    Option(
        "blend", "choice", "alpha",
        help="How the mark mixes with the base image (default: alpha)",
        choices=("alpha", "multiply", "overlay"), group="blend",
    ),
    Option(
        "ink", "float", 0.32,
        help=(
            "Darkness of the mark for multiply/overlay blend, 0-1 (default: 0.32). For "
            "--blend overlay, 0.5 is the mathematical no-op point (the mark becomes "
            "invisible) - stay clearly above or below 0.5. For multiply, smaller values "
            "give a darker mark; 1.0 is the no-op point there instead."
        ),
        min=0, max=1, group="blend",
    ),
    Option(
        "invisible", "bool", False,
        help=(
            "Also embed a hidden backup mark; see --invisible-mode for which kind. For "
            "video this only supports --invisible-mode spread (lsb can't survive "
            "re-encoding), embedded per-frame with its own key rotation - see README."
        ),
        group="invisible", scope=("image", "video"),
    ),
    Option(
        "invisible_text", "str", None,
        help="Text for the hidden mark, defaults to --text",
        group="invisible", scope=("image", "video"),
        requires=(("invisible", True),),
    ),
    Option(
        "invisible_mode", "choice", "lsb",
        help=(
            "Which hidden-mark scheme --invisible uses (default: lsb). 'lsb' is the "
            "original least-significant-bit mark: fast, needs no key, but destroyed by "
            "any recompression, resize, or format change - save as PNG; images only, "
            "video ignores --invisible entirely with 'lsb' (see --invisible-mode spread). "
            "'spread' is a block-DCT spread-spectrum mark that survives ordinary JPEG/H.264 "
            "recompression and a block-aligned crop, at the cost of needing --ss-key (or "
            "having one generated and printed for you) to read it back later, being "
            "slower, and needing a reasonably large image/frame - see README for the full "
            "tradeoffs. The only mode video supports."
        ),
        choices=("lsb", "spread"), group="invisible", scope=("image", "video"),
        requires=(("invisible", True),),
    ),
    Option(
        "ss_key", "int", None,
        help=(
            "Key for --invisible-mode spread (an arbitrary integer). Reusing it lets you "
            "decode the mark later with extract_invisible.py --ss-key; omit it and one is "
            "generated and printed for you to save. Plays the same reproducibility role "
            "--seed plays for jitter - but unlike --seed, it is NOT a "
            "secrecy/access-control mechanism: a wrong guess has roughly a 1-in-4 "
            "measured chance of still decoding the payload (a bit more for --id "
            "payloads). See README. For video, this is the BASE key: the actual "
            "per-frame key rotates from it every --jitter-refresh-seconds, same idea as "
            "the visible grid's jitter."
        ),
        group="invisible", scope=("image", "video"),
        requires=(("invisible", True), ("invisible_mode", "spread")),
    ),
    Option(
        "ss_alpha", "float", None,
        help=(
            f"Embedding strength for --invisible-mode spread (default: "
            f"{wmcore.DEFAULT_SS_ALPHA} for images, {wmcore.DEFAULT_SS_ALPHA_VIDEO} for "
            f"video - video needs more to survive real H.264 re-encoding at a normal "
            f"quality; see bench/results/P4-NOTES.md). Higher survives more aggressive "
            f"recompression and busier/noisier images at the cost of a small but real "
            f"visual imprint; lower is more invisible but more fragile."
        ),
        min=0.01, max=100, group="invisible", scope=("image", "video"),
        requires=(("invisible", True), ("invisible_mode", "spread")),
    ),
    Option(
        "stamp_id", "bool", False,
        help=(
            "Stamp this output with a short random watermark ID and record it - which file, "
            "when, the key, the settings, and the hash of the finished output - in the ledger "
            "(~/.watermark-tool/ledger.sqlite3 unless --ledger says otherwise). The same ID goes "
            "into every layer that can carry it: the hidden mark instead of the long text (with "
            "--invisible), image metadata, the C2PA manifest (with --c2pa) and, with --grid-id, "
            "the visible grid. One `extract` on a suspect copy then returns the ID and its "
            "ledger record with no key to remember (the ledger's recorded keys are tried), and "
            "the ID's check characters let a decoder reject junk. It is not a proof: anyone who "
            "can read an ID can copy it into another file (see README, 'Watermark ID & ledger')."
        ),
        flag="--id", group="id",
    ),
    Option(
        "id_value", "str", None,
        help=(
            "Use this specific ID instead of a random one (10 characters, with its check "
            "characters - one from a previous run, or from wmcore.wmid.generate()). It must not "
            "already be in the ledger, so it applies to one file; not with --batch. Mostly for "
            "reproducible tests."
        ),
        group="id", requires=(("stamp_id", True),),
    ),
    Option(
        "grid_id", "bool", False,
        help=(
            "Also print the ID in the visible grid text (after --text), so a person who sees "
            "a repost can read it straight off the picture. Works for video too - the only "
            "ID-carrying layer besides C2PA a video has until it gets a hidden mark. Makes each "
            "tile wider; the mark is a little more conspicuous and no two copies look alike."
        ),
        group="id", requires=(("stamp_id", True),),
    ),
    Option(
        "ledger", "str", None,
        help=(
            "Ledger file for --id (created if missing, mode 0600 - it stores spread-spectrum "
            "keys). Default: $WATERMARK_TOOL_LEDGER, else ~/.watermark-tool/ledger.sqlite3."
        ),
        local=True, group="id", requires=(("stamp_id", True),),
    ),
    Option(
        "author", "str", None,
        help="Author name to embed in EXIF/PNG metadata (images only)",
        group="metadata", scope=("image",),
    ),
    Option(
        "copyright_text", "str", None,
        help="Copyright string to embed in EXIF/PNG metadata (images only)",
        flag="--copyright", group="metadata", scope=("image",),
    ),
    Option(
        "adaptive", "bool", False,
        help=(
            "Content-aware placement: detect faces and other high-detail 'main subject' "
            "regions and increase watermark density (and, for --blend multiply/overlay, "
            "darkness) exactly there - where a clean inpaint-based removal is most likely "
            "to leave visible artifacts. Uses opencv's bundled face detector plus a local "
            "detail-energy pass; no extra download or dependency."
        ),
        group="adaptive",
    ),
    Option(
        "adaptive_strength", "float", 0.6,
        help=(
            "0-1, how strongly to boost density/darkness in detected regions (default: "
            "0.6). 0 behaves like --adaptive was never passed."
        ),
        min=0, max=1, group="adaptive",
        requires=(("adaptive", True),),
    ),
    Option(
        "adaptive_detail_weight", "float", 0.5,
        help=(
            "0-1, weight of the general high-detail 'main subject' signal relative to "
            "face detection, which is always full-strength wherever a face is found "
            "(default: 0.5)"
        ),
        min=0, max=1, group="adaptive",
        requires=(("adaptive", True),),
    ),
    Option(
        "face_cascade", "str", None,
        help=(
            "Path to a custom Haar cascade XML for face detection (default: the "
            "frontal-face model bundled with opencv)"
        ),
        local=True, group="adaptive",
        requires=(("adaptive", True),),
    ),
    Option(
        "adaptive_infill_divisor", "float", 2.0,
        help=(
            "How much denser the extra grid is inside detected regions - --spacing is "
            "divided by this there (default: 2.0, i.e. twice as dense)"
        ),
        min=1, max=16, group="adaptive",
        requires=(("adaptive", True),),
    ),
    Option(
        "adaptive_track", "bool", True,
        help=(
            "Video only: keep the detected sensitivity map tracking the subject/camera "
            "across the whole clip using cheap per-frame optical flow between periodic "
            "re-detections, instead of detecting once from frame 0 and reusing that "
            "snapshot for the entire video regardless of what moves (default: on). See "
            "--adaptive-redetect-seconds. Pass --no-adaptive-track for the old, "
            "detect-once behavior."
        ),
        flag="--no-adaptive-track", negate=True, group="adaptive", scope=("video",),
        requires=(("adaptive", True),),
    ),
    Option(
        "adaptive_redetect_seconds", "float", 1.0,
        help=(
            "Video only, with --adaptive-track: how often (in seconds of output) to "
            "re-run the actual face/detail detection to correct drift and catch new "
            "subjects; the sensitivity map is carried forward frame-by-frame with "
            "optical flow in between (default: 1.0)"
        ),
        min=0.1, group="adaptive", scope=("video",),
        requires=(("adaptive", True), ("adaptive_track", True)),
    ),
    Option(
        "scene_cut_detect", "bool", True,
        help=(
            "Video only: watch for hard scene cuts (a big frame-to-frame content jump) "
            "and treat one as an immediate boundary - a fresh jitter pattern and, with "
            "--adaptive-track, a fresh detection pass - right at the cut, instead of "
            "carrying the previous shot's mark and sensitivity map a few more frames "
            "into the new shot until the next timed refresh happens to land (default: "
            "on)."
        ),
        flag="--no-scene-cut-detect", negate=True, group="video", scope=("video",),
    ),
    Option(
        "scene_cut_threshold", "float", 0.12,
        help=(
            "Video only: mean per-pixel grayscale change (0-1, on a downscaled frame) "
            "above which two consecutive frames count as a scene cut (default: 0.12). "
            "Lower catches softer cuts/hard pans at the cost of more false positives on "
            "fast motion; higher misses subtler cuts."
        ),
        min=0, max=1, group="video", scope=("video",),
        requires=(("scene_cut_detect", True),),
    ),
    Option(
        "adaptive_preview", "str", None,
        help=(
            "Images only, single-file mode only: also save the detected sensitivity mask "
            "as a grayscale PNG at this path (white = most boosted), for tuning the flags "
            "above"
        ),
        local=True, group="adaptive", scope=("image",),
        requires=(("adaptive", True),),
    ),
    Option(
        "c2pa", "bool", False,
        help=(
            "Also sign the output with a C2PA (Content Credentials) manifest - a "
            "cryptographically signed hash-binding, complementary to (not a replacement "
            "for) the pixel-domain grid/invisible marks above. Unlike "
            "--invisible/--author, this applies to BOTH images and video. Needs "
            "c2pa-python + cryptography installed (pip install -r requirements.txt); see "
            "cli/c2pa_provenance.py for the full design rationale and README.md for the "
            "C2PA section."
        ),
        group="c2pa",
    ),
    Option(
        "c2pa_cert", "str", None,
        help=(
            "PEM file with your signing certificate (plus any intermediate chain certs) "
            "for --c2pa. Default: an auto-generated, reused local dev certificate at "
            "~/.watermark-tool/ - self-signed, so C2PA-aware readers report the signing "
            "credential as untrusted rather than showing a verified checkmark. Must be "
            "given together with --c2pa-key."
        ),
        local=True, group="c2pa",
        requires=(("c2pa", True),),
    ),
    Option(
        "c2pa_key", "str", None,
        help=(
            "Private key PEM matching --c2pa-cert. Must be given together with "
            "--c2pa-cert."
        ),
        local=True, group="c2pa",
        requires=(("c2pa", True),),
    ),
    Option(
        "c2pa_alg", "choice", c2pa_provenance.DEFAULT_ALG,
        help=(
            f"Signing algorithm for --c2pa (default: {c2pa_provenance.DEFAULT_ALG}). Only "
            f"relevant with --c2pa-cert/--c2pa-key - the auto-generated dev certificate is "
            f"always {c2pa_provenance.DEFAULT_ALG} regardless of this flag."
        ),
        choices=("es256", "es384", "es512", "ed25519", "ps256", "ps384", "ps512"), group="c2pa",
        requires=(("c2pa", True),),
    ),
    Option(
        "c2pa_title", "str", None,
        help=(
            "Title recorded in the C2PA manifest for --c2pa (default: the output "
            "filename)"
        ),
        group="c2pa",
        requires=(("c2pa", True),),
    ),
)

# Read-back options (extract_invisible.py; its positional `image` argument is the CLI's own).
EXTRACT_OPTIONS = (
    Option(
        "mode", "choice", "auto",
        help=(
            "Which invisible-mark format to look for. 'auto' (default) tries LSB first, "
            "then spread-spectrum if a key is available (--ss-key, or the keys in the "
            "ledger) and LSB found nothing or found something that isn't a watermark ID, "
            "then - only in auto - the ID in the file's metadata and C2PA manifest. This "
            "matches --invisible-mode lsb being watermark.py's own default, while still "
            "working automatically against a spread-spectrum mark once a key is known."
        ),
        choices=("lsb", "spread", "auto"),
    ),
    Option(
        "ss_key", "int", None,
        help=(
            "Key for a spread-spectrum mark (enables trying it in --mode auto too). Must "
            "match the --ss-key used to embed it. Optional when the ledger has keys: "
            "--mode spread then tries each recorded key, newest first."
        ),
    ),
    Option(
        "jitter_refresh_seconds", "float", 4.0,
        help=(
            "Video only: must match the --jitter-refresh-seconds used to embed the "
            "invisible mark (default: 4.0, its own default too) - it decides where the "
            "per-frame key derived from --ss-key changes. Not a secret, unlike --ss-key "
            "itself, but has to agree for the segment math to land on the same key."
        ),
        min=0, group="video", scope=("video",),
    ),
    Option(
        "ledger", "str", None,
        help=(
            "Ledger file to look a found watermark ID up in, and to take spread-spectrum "
            "keys from when --ss-key isn't given. Default: $WATERMARK_TOOL_LEDGER, else "
            "~/.watermark-tool/ledger.sqlite3 - used only if it already exists (extract "
            "never creates one)."
        ),
        local=True,
    ),
    Option(
        "use_ledger", "bool", True,
        help="Don't consult any ledger: no ID lookup, no recorded keys tried",
        flag="--no-ledger", negate=True,
    ),
    Option(
        "max_length", "int", MAX_PLAUSIBLE_LENGTH,
        help=(
            f"Longest payload (bytes) to consider plausible while scanning (default: "
            f"{MAX_PLAUSIBLE_LENGTH})"
        ),
        min=1,
    ),
    Option(
        "layers", "bool", False,
        help=(
            "Also print what EACH layer (LSB, spread-spectrum, metadata, C2PA) found on its own - "
            "found / nothing readable / not checked and why - instead of only the first layer "
            "that gave an answer. Every layer is always checked either way and the per-layer "
            "result is always in the Python/JSON result; this only adds the lines to the text "
            "report. (If the layers disagree about the ID, that warning is printed regardless.)"
        ),
    ),
    Option(
        "rescale", "bool", False,
        help=(
            "For a spread-spectrum mark: if the plain decode finds nothing, also search "
            "for a resize applied after embedding (uniform scale, top-left corner "
            "unchanged; range set by --rescale-range). Slower - a few seconds per image, "
            "more when nothing is found."
        ),
    ),
    Option(
        "rescale_range", "float_pair", tuple(wmcore.SS_RESCALE_RANGE),
        help=(
            f"Decode-time resize factors --rescale searches between (default: "
            f"{wmcore.SS_RESCALE_RANGE[0]:g} {wmcore.SS_RESCALE_RANGE[1]:g}). An image "
            f"that was shrunk to 75% needs 1.333 to recover, so 0.5-2.0 covers images that "
            f"were resized anywhere between 0.5x and 2x."
        ),
        metavar=("MIN", "MAX"),
    ),
    Option(
        "resync", "bool", True,
        help=(
            "On by default. When a spread-spectrum ID could not be read the plain way, look for it again as "
            "if the copy had been CROPPED: re-find where the 8x8 block grid it was stamped on starts "
            "(a screenshot trimmed to the picture almost never starts on a multiple of 8 pixels) and how the "
            "rows line up (trimming the left or right edge changes the row length), also for a mirrored copy. "
            "Cheap (tens of ms per key; for a video it runs on every frame the plain decode misses, unmirrored only). "
            "Also covers mirrored copies (the plain mirrored decodes are skipped while it is on) and, with --rescale, "
            "a copy that was cropped AND resized. IDs only. Pass --no-resync to skip it."
        ),
        flag="--no-resync", negate=True,
    ),
    Option(
        "resync_budget_seconds", "float", wmcore.DEFAULT_RESYNC_BUDGET_SECONDS,
        help=(
            f"Wall-clock budget for the crop re-alignment search (default: "
            f"{wmcore.DEFAULT_RESYNC_BUDGET_SECONDS:g}s). It only matters with a very long ledger or a huge "
            f"image; running out turns a real mark into 'inconclusive', never into a wrong ID."
        ),
    ),
    Option(
        "rescale_budget_seconds", "float", wmcore.DEFAULT_RESCALE_BUDGET_SECONDS,
        help=(
            f"Wall-clock budget for --rescale's search (default: "
            f"{wmcore.DEFAULT_RESCALE_BUDGET_SECONDS:g}s). Pass a negative number or 0 for "
            f"effectively no search, or a very large value / use the Python API with "
            f"max_seconds=None to search exhaustively regardless of how long it takes. "
            f"Hitting the budget can only turn a real mark into a 'not found' - it never "
            f"accepts a weaker result."
        ),
    ),
)

SPEC_OPTIONS = tuple(o for o in OPTIONS if o.role == "spec")
RUN_OPTIONS = tuple(o for o in OPTIONS if o.role == "run")
_WM_BY_NAME = {o.name: o for o in SPEC_OPTIONS}
_EX_BY_NAME = {o.name: o for o in EXTRACT_OPTIONS}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check_value(opt, value):
    """Validate (and lightly normalise) one value against its option; returns
    the value to store. Raises SpecError with the option's name in the message."""
    label = opt.name
    if value is None:
        if opt.required:
            raise SpecError(f"{label} is required")
        return None
    t = opt.type
    if t == "bool":
        if not isinstance(value, bool):
            raise SpecError(f"{label}: expected true or false, got {value!r}")
        return value
    if t == "int":
        if not _num(value) or (isinstance(value, float) and not value.is_integer()):
            raise SpecError(f"{label}: expected an integer, got {value!r}")
        value = int(value)
    elif t == "float":
        if not _num(value) or not math.isfinite(value):
            raise SpecError(f"{label}: expected a number, got {value!r}")
        value = float(value)
    elif t == "str":
        if not isinstance(value, str):
            raise SpecError(f"{label}: expected text, got {value!r}")
        if value == "":
            if opt.required:
                raise SpecError(f"{label} is required")
            return None
        return value
    elif t == "choice":
        if value not in opt.choices:
            raise SpecError(f"{label}: must be one of {', '.join(opt.choices)}, got {value!r}")
        return value
    elif t == "float_pair":
        if not (isinstance(value, (list, tuple)) and len(value) == 2
                and all(_num(v) and math.isfinite(v) for v in value)):
            raise SpecError(f"{label}: expected two numbers, got {value!r}")
        return (float(value[0]), float(value[1]))
    if opt.min is not None and value < opt.min:
        raise SpecError(f"{label}: must be at least {opt.min:g}, got {value:g}")
    if opt.max is not None and value > opt.max:
        raise SpecError(f"{label}: must be at most {opt.max:g}, got {value:g}")
    return value


# ---------------------------------------------------------------------------
# Typed request objects
# ---------------------------------------------------------------------------

def _make_spec_class(clsname, options, by_name, doc):
    flds = []
    for o in options:
        flds.append((o.name, Any) if o.required else (o.name, Any, field(default=o.default)))

    def __post_init__(self):
        for o in options:
            setattr(self, o.name, check_value(o, getattr(self, o.name)))

    @classmethod
    def from_mapping(cls, data, *, allow_local=True):
        """Build from a plain dict (JSON body, config file...). Unknown keys are
        errors - a typo like "blnd" must not silently fall back to a default.
        With allow_local=False (what the HTTP adapter uses) any option that
        names a path on this machine is refused."""
        unknown = sorted(set(data) - set(by_name))
        if unknown:
            raise SpecError("unknown option(s): " + ", ".join(unknown))
        missing = [o.name for o in options if o.required and o.name not in data]
        if missing:
            raise SpecError(", ".join(missing) + " is required")
        if not allow_local:
            bad = sorted(k for k, v in data.items() if by_name[k].local and v not in (None, ""))
            if bad:
                raise SpecError(", ".join(bad) + ": not available over this interface "
                                "(it names a path on the machine running the code)")
        return cls(**data)

    def to_dict(self):
        return {o.name: (list(getattr(self, o.name)) if isinstance(getattr(self, o.name), tuple)
                         else getattr(self, o.name)) for o in options}

    def replace(self, **changes):
        return _dc_replace(self, **changes)

    ns = {"__post_init__": __post_init__, "from_mapping": from_mapping, "to_dict": to_dict,
          "replace": replace, "__doc__": doc, "OPTIONS": tuple(options)}
    cls = make_dataclass(clsname, flds, namespace=ns)
    cls.__module__ = __name__  # so instances pickle by reference (batch workers, tests)
    return cls


WatermarkSpec = _make_spec_class(
    "WatermarkSpec", SPEC_OPTIONS, _WM_BY_NAME,
    "Everything that determines how ONE file is watermarked. Fields and defaults are "
    "generated from OPTIONS; construct directly, or from a dict with from_mapping().",
)
ExtractSpec = _make_spec_class(
    "ExtractSpec", EXTRACT_OPTIONS, _EX_BY_NAME,
    "How to look for a hidden payload. Fields and defaults are generated from EXTRACT_OPTIONS.",
)


# ---------------------------------------------------------------------------
# Derived interfaces
# ---------------------------------------------------------------------------

def add_arguments(parser, options):
    """Add the argparse flags for `options` to `parser`, in schema order, with
    exactly the flags/defaults/help the CLIs had before the schema existed."""
    for o in options:
        kw = {"dest": o.name, "help": o.help.replace("%", "%%")}
        if o.type == "bool":
            kw["action"] = "store_false" if o.negate else "store_true"
        elif o.type == "float_pair":
            kw.update(type=float, nargs=2, metavar=o.metavar, default=list(o.default))
        else:
            if o.type in ("int", "float"):
                kw["type"] = int if o.type == "int" else float
            kw["default"] = o.default
            if o.choices:
                kw["choices"] = list(o.choices)
        if o.required:
            kw["required"] = True
        parser.add_argument(o.cli_flag, **kw)


def spec_from_namespace(cls, ns):
    """WatermarkSpec/ExtractSpec from an argparse Namespace."""
    return cls(**{o.name: getattr(ns, o.name) for o in cls.OPTIONS})


def schema(*, include_local=True):
    """JSON-serialisable description of every option (served at /api/schema)."""
    def dump(opts):
        return [o.to_dict() for o in opts if include_local or not o.local]
    return {
        "version": 1,
        "groups": [{"id": k, "title": v} for k, v in GROUPS.items() if k != "run"],
        "watermark": dump(SPEC_OPTIONS),
        "extract": dump(EXTRACT_OPTIONS),
    }
