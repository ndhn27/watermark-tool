# Mộc — Diagonal Grid Watermark

A small tool that stamps images **and video** with a **tiled, diagonal grid
watermark** instead of a single corner logo. Comes in two flavors:

- **`cli/watermark.py`** — a Python script for images and video, single
  files or whole folders, parallelized across CPU cores with a progress bar.
- **`web/index.html`** — a single-file, no-build browser tool with live
  preview and multi-image batch export (nothing is uploaded anywhere; it
  all runs on-canvas in your browser).

## Why a grid instead of a corner logo?

A logo tucked in one corner is gone the moment someone crops it out, and
small, localized AI inpainting can erase it in one pass. A watermark tiled
diagonally across the *entire* frame is a different problem to remove:
getting rid of it cleanly means repainting almost the whole image (or every
frame of a video), which is far more work and tends to leave visible
artifacts or blur — especially over detailed areas like faces or text.

## Removal-resistance features (CLI)

A perfectly regular grid still has weaknesses: it has a strong signature in
the frequency domain (FFT) that can be estimated and subtracted, especially
across many images stamped with the same pattern, and a flat white layer can
in principle be separated from the base image. The CLI addresses both:

- **Per-tile jitter** — every tile gets its own random position offset,
  opacity, and rotation, derived from `--seed` (reuse the same seed to
  reproduce an identical pattern; omit it for a fresh one each run). This
  breaks the grid's periodicity so it's much harder to model out.
  **Video gets this refreshed over time, too** (`--jitter-refresh-seconds`,
  default `4.0`): the jitter pattern is regenerated every N seconds of
  output instead of one static pattern for the whole clip. A single
  unchanging mark stamped over moving footage is itself a collusion/
  averaging-attack target — the base content moves frame to frame but the
  mark wouldn't, so a per-pixel median across enough frames estimates and
  subtracts it, no *separate* files required the way the paragraph below
  otherwise assumes. Pass `0` to opt back into one static pattern for the
  whole video (only useful for exact reproducibility with older output).
  **Performance note:** each refresh rebuilds the whole worker pool, which
  has a real, roughly fixed cost per rebuild — so a short interval adds up
  fast on a short clip (rebuild overhead is a bigger fraction of total
  runtime) and gets worse with more `--workers` (more worker processes to
  spawn each time). Measured on one machine: a 6s clip took 1.82s at the
  `4.0` default vs. 6.29s at `0.3` (~3.5x); the tool prints a note if you
  set it under 1 second. Benchmark your own setup before committing to a
  very short interval for a long/production run, and only go that short if
  your threat model genuinely calls for defending against an adversary who
  can median/average frames over a window that narrow.
  **Scene cuts refresh it early, too** (`--scene-cut-detect`, on by
  default): a hard cut to a different shot forces the jitter pattern to
  regenerate right there instead of drifting a few more frames into the new
  shot before the next timed refresh happens to land — see
  `--scene-cut-threshold` in the table below. Pass `--no-scene-cut-detect`
  to go back to a purely time-based refresh.
- **Blend modes** (`--blend multiply|overlay`) — instead of a flat
  alpha-over, the mark mixes into the base image's own pixel values
  (`--ink` controls how strongly), so it can't be cleanly separated as its
  own layer. Applies to both images and video frames.
  **Watch `--ink` with `--blend overlay`:** with the overlay formula used
  here, `--ink 0.5` is the exact mathematical no-op — it leaves every pixel
  unchanged, so the watermark is completely invisible, not just "medium
  strength." This is easy to hit by accident if you assume 0.5 is a safe
  middle value (it's the *0.32 default* that's a safe middle value). Stay
  clearly above or below 0.5 for a visible mark. Multiply doesn't have this
  trap in the middle of the range — there, smaller `--ink` means a darker
  mark and the no-op point is instead at the top of the range (`--ink 1.0`).
- **Invisible backup mark** (`--invisible`) — readable back
  with `cli/extract_invisible.py`, in one of two schemes
  (`--invisible-mode`); `spread` also works for video (see [Video
  support](#video-support) for what's different there), `lsb` is images only:
  - **`lsb`** (default) — a simple LSB (least significant bit) payload
    spread across the image. Lightweight, needs no key, but does **not**
    survive recompression, resizing, or screenshots — it can confirm
    provenance on a losslessly-saved (PNG) copy even if the visible grid
    gets cropped or painted over, nothing more.
  - **`spread`** — a block-DCT spread-spectrum payload (classic
    spread-spectrum watermarking — Cox et al. 1997 — adapted to carry a
    multi-byte payload). This one **does** survive ordinary JPEG
    recompression (tested clean down to quality 50) and a block-aligned
    crop, because the payload lives in mid-frequency DCT coefficients of
    each 8x8 block instead of pixel LSBs, and decoding sums the signal
    across every surviving repeat before making a bit decision (the same
    "processing gain" that makes spread-spectrum radio work). It needs
    `--ss-key` (an arbitrary integer; one is generated and printed for
    you if you don't pass one) to read the mark back, works on JPEG
    output where `lsb` cannot, is noticeably slower, and needs a
    reasonably large image (very small images may not have enough 8×8
    blocks to carry even one copy of the payload).
    **`--ss-key` is a reproducibility value, like `--seed` — it is *not*
    a secrecy or access-control mechanism.** Unlike `--seed`, this gap is
    not theoretical: because every block spreads its bit against the same
    fixed sequence (needed so a crop doesn't invalidate it — see below)
    and decoding sums that correlation across every repeat in the image, a
    wrong key whose sequence happens to have even a modest chance overlap
    with the real one gets that overlap amplified by the same combining
    step that makes the real key so reliable. **Measured on real images, a
    uniformly random wrong key has roughly a 1-in-4 chance of still
    decoding the payload correctly.** (That figure is for text payloads;
    the ID-aware reader used for `--id` marks leaks somewhat more — see
    [Watermark ID & ledger](#watermark-id--ledger---id); and with the crop re-alignment search on, a
    wrong key reads the ID far more often, 54 of 60 in the sample in `bench/results/CROP-NOTES.md` §5.) Never treat this
    as a way to keep a payload from being read by someone who doesn't have
    the key.
    **Resized images (`extract_invisible.py --rescale`).** The mark is
    anchored to the original image's 8×8 block grid, so a resized copy
    doesn't decode as-is. `--rescale` searches for the resize that was
    applied and undoes it before decoding: a cheap sweep over scale factors
    on a small top-left patch finds where the block correlations become
    strongly bimodal, refines that on bigger patches, and only then tries
    full-image decodes (a third, pricier sweep tier kicks in as a last
    resort on images large enough to afford it — roughly 1300px+ on the
    short side). It is opt-in because it costs a few seconds per
    image (more when nothing is found), and it has real limits:
    uniform scale only (aspect ratio kept), the top-left corner must still be
    the top-left corner (a resize combined with a crop is not searched for —
    measured directly in a pilot bench run simulating that exact repost
    pattern: resize + JPEG + an unaligned crop chained in one pass, 0/3
    recovered with `--rescale` even given the full default 20s budget and
    confirmed not budget-starved, see `bench/results/p0-new-attacks/` and
    `bench/README.md`'s Attacks section), and heavy downscaling (~0.5×,
    worse with JPEG on top) can erase the mark
    outright, in which case there is nothing to find. A search this wide
    over a noisy statistic also needs guardrails against false confidence
    and unbounded runtime, so the result carries a **confidence score**
    (`--rescale`'s output prints it; the Python API returns it on
    `RescaleMatch.confidence`) — a 0–1 heuristic index built from how
    strongly the winning scale stood out, how many payload repeats backed
    it up, and how many independently tried widths agreed, **not** a
    calibrated probability — and the search runs under a **wall-clock
    budget** (`--rescale-budget-seconds`, default 20s; the Python API's
    `max_seconds`), so a stubborn "nothing here" image can't hang
    indefinitely. Hitting the budget only ever makes the search give up
    sooner — it's reported separately (as "inconclusive", not "not found")
    and can never turn a marginal decode into an accepted one. Two measured
    details explain why it is not a short list of scale factors: the
    resample has to land within ~1-2 px of the original width (a fixed
    list like
    0.5/0.75/1.25/2.0 only hit 4 of 13 test resizes), and a decode a few px
    off can validate as a junk 1-byte payload, so a result is only accepted
    if it is printable text that decodes identically at two different widths.
    On 36 synthetic test images (1280×960 to 1920×1080, resized 0.5×-2× with
    bilinear/bicubic/Lanczos, plain or JPEG 85/75) it recovered 34, i.e.
    every one that decoding at the known original size also recovers (the
    other two had lost the mark), with no wrong payloads. Those are synthetic
    images, not a survey of real photos. **The cheap scale-sweep statistic
    that drives the search *is* corner-anchored** (only the final decode step
    scans the whole image, as the next paragraph says) — at some image
    size/content combinations that corner is a bad spot to judge from and the
    search fails even though the mark is intact; see `bench/results/P0-NOTES.md`
    §1 for a worked example (a visible-grid interaction, not specific to
    resizing itself).

  Both schemes tile the payload back-to-back across the whole image (across
  pixels for `lsb`, across 8×8 blocks for `spread`) so a repeat survives a
  crop or paint-over of part of the image, and both extractors scan for a
  surviving repeat wherever it lands rather than only trusting the
  top-left corner. `lsb`'s extractor only trusts a repeat once a second
  copy elsewhere confirms it (falling back to a single uncorroborated
  match only if that's genuinely all that's left); `spread`'s extractor
  instead *combines* every repeat it can find via the correlation math
  above before deciding, which tolerates real per-repeat noise far better
  than requiring an exact match would. Both share one structural
  limitation from tiling in raster order: a crop that changes the image's
  *width* (not just height), or one that starts off the 8-pixel grid, shifts every repeat's position
  within the flattened sequence, so the plain decoders above do not read it. **`extract` now handles that
  case separately** (`--resync`, on by default — see "Crop resync" below), for watermark IDs. A pilot
  bench run also saw `spread`'s plain
  decoder occasionally return a **confident wrong string** (not just fail
  to find one) after a horizontal flip — only 1 misdecode in 8 trials, too
  small a sample to quote a rate from, but worth knowing a flip's failure
  mode isn't always a clean "nothing found" (see
  `bench/results/p0-new-attacks/report.md`). The same showed up with an ID
  payload: 28 wrong strings in 1968 synthetic trials — 24 on left crops and
  pixel-unaligned crops (outside the documented range), plus 2 on the supported
  block-aligned top crop, 1 on a flip and 1 at JPEG q30 (`bench/results/P2-NOTES.md`);
  the ID decoder below returned none of them.

  **Mirrored copies (left-right, top-bottom, or both).** A mirror reverses the
  block order and reflects every block, so the plain decoder reads noise -
  but the mark is not damaged (oracle bit error rate 0.000 on the mirrored
  copies that have an oracle; plain detection 0/240), because mirroring the
  suspect image back restores the original grid exactly. `extract` therefore
  also tries the three mirrorings (`wmcore/orient.py`) and says so: "recovered
  after flipping the image back". **A watermark ID is read this way, a
  free-text payload is not:** the ID rule (a check-valid 10-character ID) has a
  false-accept chance of about 1e-10 per orientation, while the text decoder's
  rule is far looser, and on periodic never-stamped test images it returned
  junk more often the more orientations it was run on - so mirrored text
  payloads are deliberately not searched. Measured on synthetic images and on
  7 classic test photos: IDs read on 240/240 and 35/35 mirrored copies (plain
  decoder: 0/240 and 0/35), no extra false positives (0/400 and 0/28 unmarked
  images), no wrong string on an ID payload (`bench/results/P3-NOTES.md`).
  Not covered: a copy that was mirrored *and* resized (the resize search runs
  on the image as given: 0/16 before, unchanged), 90-degree rotations and
  transposes, and the `lsb` layer. Cost: about 3-4x the plain decode's time on
  an image with no mark in it; a marked, unmirrored ID is answered by the
  first attempt.
- **Content-adaptive placement** (`--adaptive`, CLI only — no equivalent
  toggle in `web/index.html`) — a lightweight pass (opencv's bundled Haar
  face detector, plus a local detail-energy signal that also catches other
  high-detail "main subject" areas: patterned clothing, foreground
  objects, on-image text) finds the regions of the frame where a clean
  inpaint-based removal is most likely to leave a visible seam, and stamps
  a second, denser grid on top exactly there (`--adaptive-infill-divisor`,
  default `2.0`, sets how much denser).
  For `--blend multiply|overlay`, it also darkens the mark further in
  those same regions (`--adaptive-strength`, default `0.6`, 0-1) — moving
  `--ink` further from that mode's own no-op point, the same one described
  above, never past it. Outside the detected regions the image is
  identical to the non-adaptive path — the boost is additive, not a
  global change. No extra model download or dependency: the face detector
  ships inside `opencv-python-headless`, already required for video.
  `--adaptive-preview some.png` (single images only) saves the detected
  mask itself (white = most boosted) so you can sanity-check or tune
  `--adaptive-detail-weight` (how much weight the general detail signal
  gets relative to face detection, default `0.5`) before committing to a
  batch run. `--face-cascade` swaps in a different Haar cascade XML if the
  bundled frontal-face one isn't a good fit for your subjects (e.g. a
  profile-face cascade for photos that aren't mostly front-facing).
  **Video:** by default (`--adaptive-track`, on) the mask is kept current
  across the whole clip instead of frozen at frame 0. wmcore.AdaptiveTracker
  runs once per decoded frame, entirely in the main process, separately
  from the worker pool that does the actual frame compositing: most frames
  it just carries the last detected mask forward with cheap optical flow on
  a downscaled copy (a moving subject or a panning/zooming camera is what
  this is for), and periodically (`--adaptive-redetect-seconds`, default
  `1.0`) it re-runs the real face/detail detection to correct the drift
  flow accumulates and pick up new subjects entering frame. A detected
  scene cut (see `--scene-cut-detect` above) skips the flow-warp for that
  transition and forces a fresh detection instead, since flow computed
  across a hard content change describes noise, not motion. Whatever the
  tracker currently holds when a `--jitter-refresh-seconds` segment
  boundary is reached is what that segment's mask gets built from, so the
  boost stays close to what's actually on screen for that segment rather
  than a fossil of frame 0. This adds no per-frame cost to the worker
  pool - only the once-per-segment mask rebuild that was already there.
  Pass `--no-adaptive-track` for the old behavior (detect once from frame
  0, reuse that snapshot for the entire video) if you need exact
  reproducibility with older output, or want to skip the extra per-frame
  optical-flow cost on a static/tripod shot where it buys nothing.
  **Scope:** `--adaptive` only changes the *visible* grid layer (its
  density and, for multiply/overlay, its darkness). It has no effect on
  either invisible mark - `lsb` still requires a lossless save and is
  destroyed by any recompression/resize, and `spread` keeps exactly the
  same JPEG-recompression/crop robustness and `--ss-key` caveats described
  above either way. The biggest limitation of the invisible marks isn't
  changed by turning `--adaptive` on.
- **Provenance metadata** (`--author`, `--copyright`, images only) —
  written into real EXIF (JPEG, via `piexif`) or PNG text chunks, so tools
  that read standard metadata see attribution even without looking at the
  pixels. Anyone can strip metadata, so treat this as a complementary
  signal, not the main defense.
- **C2PA / Content Credentials** (`--c2pa`, images **and** video) — a
  cryptographically signed manifest, a fundamentally different kind of
  signal from everything else on this list: not a pattern to survive an
  edit, but a signature that a specific edit history produced this exact
  file. See the dedicated [C2PA (Content Credentials)](#c2pa-content-credentials)
  section below for what it actually proves, the self-signed-by-default
  caveat, and full CLI usage.
- **Watermark ID & ledger** (`--id`) — one short ID per output file, written
  into every layer that can carry it and recorded in a local SQLite ledger,
  so `extract` on a suspect copy returns the ID and what it means with no
  key to remember. Not a proof and not a robustness gain; see
  [Watermark ID & ledger](#watermark-id--ledger---id).

## Threat model

Worth being explicit about what "removal-resistant" means here, because
it means different things depending on who's trying to remove the mark.

**What this defends against — casual / incidental removal.** Someone
crops the image, paints over part of it, takes a screenshot, re-uploads
through a platform that recompresses it, or just doesn't notice a
single corner logo. This is the everyday case the grid, the jitter, the
blend modes, content-adaptive placement, and the invisible backup marks
(`lsb` and `spread`) are built for, and it's a realistic bar to clear:
most removal in the wild is casual, not adversarial.

**What this does *not* defend against — a motivated adversary who knows
the scheme.** Per-tile jitter specifically defeats naive frequency-domain
estimation (FFT-averaging a fixed periodic pattern across many stamped
images), but that's one specific attack, not immunity in general. No
publicly known image watermarking technique — DCT/DWT, spread-spectrum
(this tool's own `spread` mode included), or learned marks like
HiDDEN/StegaStamp, including ones baked directly into a diffusion
model's seed (Tree-Ring, Gaussian Shading) — reliably survives someone
who deliberately targets it with tools like:

- **Regeneration / diffusion-purification attacks** — encode the image,
  add noise, and let a diffusion model (or VAE) resynthesize it; the
  reconstruction preserves perceptual content but not a low-amplitude
  watermark signal sitting off the "natural image" manifold. Zhao et al.
  gave a provable treatment of this and showed it empirically breaking
  several watermarking schemes (*"Invisible Image Watermarks Are
  Provably Removable Using Generative AI"*, NeurIPS 2024). Follow-up work
  extends the same idea to marks embedded in a diffusion model's own
  seed/latent rather than pixel space.
- **Collusion / averaging attacks** — with several images carrying a
  correlated pattern, simply averaging them out can estimate and
  subtract a "robust" watermark; this is the pixel-domain reason per-tile
  jitter matters here.
- **Adaptive / white-box optimization** — with access to the detector
  (or a good enough guess at the embedding scheme), directly optimizing
  a small perturbation to flip detection.
- **Desynchronization** — rotation, scaling, or local warping breaks
  correlation-based detectors unless they resynchronize first; a
  classical weakness predating any of the generative-AI attacks above.

`WAVES` (An et al., ICML 2024) standardizes exactly this kind of stress
test and found vulnerabilities in several widely-used watermarking
methods once regeneration and adversarial attacks were included
alongside ordinary distortions.

Provenance metadata (`--author`/`--copyright`) and manifest-style
schemes like C2PA (`--c2pa` — see the
[dedicated section](#c2pa-content-credentials) below) sidestep the
pixel-robustness arms race entirely, but answer a different question —
"does this exact file match something signed" rather than "is a mark
still recoverable after edits" — so a full regeneration just produces
an unsigned image rather than a false match, which is a different
(arguably more honest) failure mode.

Bottom line: treat every mechanism in this tool as raising the cost of
casual removal, not as cryptographic proof against a determined,
resourced remover — that's still an open research problem industry-wide,
not a gap specific to this project.

## C2PA (Content Credentials)

`--c2pa` signs the output with a [C2PA](https://c2pa.org/) manifest — a
cryptographically signed record binding a claim ("this file, watermarked,
edited from that parent file, by this action") to the file's actual bytes.
It's genuinely different from every other mechanism in this tool (see the
callout in [Removal-resistance features](#removal-resistance-features-cli)
above and the [Threat model](#threat-model) section): the grid, jitter, and
invisible marks all ask "did a recoverable trace of the pattern survive
whatever happened to this image", a pixel-robustness question with no
hard guarantee. C2PA instead asks "does this exact file's content match a
manifest that was cryptographically signed at edit time" — editing the
pixels at all invalidates the signature rather than maybe surviving it, and
because it applies to the file as a signed data structure rather than to
pixel patterns, it works identically for **video**, not just images (unlike
`--author`/`--copyright`'s own file-format-metadata role, and unlike
`--invisible-mode lsb`, all image-only — `--invisible-mode spread` does
work for video, see [Video support](#video-support)).

```bash
# sign with the auto-generated local dev certificate (zero setup)
python3 cli/watermark.py in.jpg out.jpg --text "@yourhandle" --c2pa

# same, plus an author recorded in the manifest's CreativeWork assertion
# (works for video too, even though --author's OWN EXIF/PNG-metadata role
# elsewhere in this tool stays image-only)
python3 cli/watermark.py in.mp4 out.mp4 --text "@yourhandle" --c2pa --author "Your Name"

# read the manifest back
python3 cli/c2pa_verify.py out.jpg
python3 cli/c2pa_verify.py out.jpg --json   # full parsed detail
```

**Read this before showing a `--c2pa` file to anyone else.** By default,
signing uses a **local, self-signed certificate**, auto-generated on first
use and reused after that at `~/.watermark-tool/dev_cert.pem` (main() does
this once up front, not per file, so a `--batch` run's worker processes
all reuse the same identity rather than racing to generate their own). The
resulting signature is completely real — the hash-binding and signature
checks genuinely pass, and `c2pa_verify.py`'s `Validation state: Valid` on
it isn't a lie — but because nothing vouches for *the certificate itself*,
every C2PA-aware reader (`c2pa_verify.py` included) reports the signing
credential as **untrusted** rather than showing a verified checkmark:

```
Validation state: Valid
  - FAILED signingCredential.untrusted: signing certificate untrusted (expected for the auto-generated dev cert - see cli/c2pa_provenance.py / README.md)
```

That failure code is the *entire* gap between "real signature, unknown
signer" and a verified badge — there is no additional "is it actually
tampered with" check hiding behind it. To close that gap, get a
certificate from a CA in the [C2PA-approved trust
list](https://opensource.contentauthenticity.org/docs/c2pa-conformance/trust-lists)
and pass it in:

```bash
python3 cli/watermark.py in.jpg out.jpg --text "@yourhandle" \
    --c2pa --c2pa-cert your_chain.pem --c2pa-key your_key.pem --c2pa-alg es256
```

`--c2pa-cert`/`--c2pa-key` must be given together. `--c2pa-alg` (default
`es256`) only matters with your own cert — the auto-generated dev
certificate is ES256-only, and a different `--c2pa-alg` without
`--c2pa-cert`/`--c2pa-key` is ignored (with a note) rather than silently
mismatched against it. `--c2pa-title` sets the manifest's recorded title
(default: the output filename).

**What's actually in the manifest:** if the input file the watermark was
applied to still exists on disk at signing time, the manifest is built as
an **EDIT** with a `parentOf` ingredient pointing at it — which makes the
C2PA side auto-generate a `c2pa.opened` action (a hash-link back to that
parent) ahead of the explicit `c2pa.watermarked` action this tool adds
(a real, registered C2PA action ID, not a made-up one) — so a reader sees
an actual two-step edit history, not just an unqualified claim. Without a
locatable parent, it falls back to a plain CREATE. `--author`, when given,
becomes a `stds.schema-org.CreativeWork` assertion inside the signed
manifest (separate from, and not a replacement for, `--author`'s own
EXIF/PNG-metadata writing for images).

**Format support and failure handling:** C2PA signing covers JPEG, PNG,
and MP4/MOV/AVI/M4V, but **not every format this tool otherwise supports**
— `.bmp`, `.mkv`, and `.webm` aren't in c2pa-python's supported list. If
you ask for `--c2pa` on one of those, signing fails with a printed note
(*"C2PA signing failed for ...; the watermark itself was still saved
without a C2PA manifest"*) rather than losing the watermark work already
done or crashing a `--batch` run over it — same graceful-degradation shape
as a missing `piexif` install elsewhere in this tool. No timestamp
authority is contacted by default (that would mean a network call on every
signing operation, with a third-party service as a new point of failure);
see `cli/c2pa_provenance.py`'s module docstring if you want to wire one up
yourself for long-term timestamp-backed validity.

Needs `c2pa-python` and `cryptography` installed (`pip install -r
requirements.txt` — both are listed there, commented, since they're
opt-in). `--c2pa` fails fast with a clear message up front if they're
missing, rather than silently skipping the manifest the way an optional
feature elsewhere in this tool might.

**Version matters: use `c2pa-python` 0.37.3 – 0.37.10** (`requirements.txt` pins it). Newer releases
(0.37.12, 0.38.0) reject the `c2pa.watermarked` action this tool writes — they require a soft-binding assertion for a watermark
action, which the tool does not write yet — so with them `--c2pa` reports that the file was not signed, and 0.37.0 / 0.37.1 raise a
`TypeError` in the signer. Measured: the C2PA tests pass on 0.37.3, 0.37.6, 0.37.9 and 0.37.10 and fail on 0.37.1, 0.37.12 and 0.38.0
(0.37.2, 0.37.4, 0.37.5, 0.37.7, 0.37.8 not tried). Writing the soft-binding assertion, so that a current release works, is open.

## Watermark ID & ledger (`--id`)

Every layer of this tool dies in a different way (recompression, a resize, a crop, a re-encode, a
stripped header), which is why they are stacked. Until `--id`, each layer carried whatever text you
gave it, and reading the `spread` mark back needed the exact `--ss-key` — which nothing stored (the
CLI printed it once, "SAVE THIS"). `--id` fixes both:

- **One short ID per output file** — 10 characters (Crockford base32: 8 random characters = 40 bits,
  plus 2 check characters), shown as `K3M9-QXR2-T4`. The same ID goes into every layer that can carry
  it: the hidden mark (in place of the long text; `--invisible`), the image metadata (PNG text chunk
  `WatermarkID`, JPEG EXIF `ImageUniqueID`), the C2PA manifest (schema.org `identifier`, `wmid:…`;
  `--c2pa`), and — only if you ask, `--grid-id` — the visible grid.
- **A ledger** — one SQLite file, `~/.watermark-tool/ledger.sqlite3` (or `--ledger PATH`, or
  `$WATERMARK_TOOL_LEDGER`), created `0600` because it stores spread-spectrum keys. One row per ID: the
  input and output file names, the SHA-256 of the input and of the **finished** output (after metadata
  and C2PA), the hidden-mark scheme, key and strength, which layers carry the ID, and the settings. The
  row is written *before* the file is stamped (`pending`) and completed afterwards, so a crash can't
  release an image whose ID the ledger has never heard of.
- **`extract` needs nothing but the file.** With no `--ss-key` it tries every key the ledger recorded
  (newest first) and accepts only a well-formed ID. It then reports the ledger record and whether the
  file is **byte-identical** to the recorded output. The `spread` layer is read by an ID-aware decoder
  first (below), then by the text decoder, so text payloads read back exactly as before.
- **`extract` checks every layer, and can tell you what each one found.** In `auto` mode it reads the
  LSB mark, the spread-spectrum mark, the metadata and the C2PA manifest *each on its own*, instead of
  stopping at the first that gives an ID. The headline answer (`found`, `text`, `scheme`, the lines it
  prints) is picked exactly as before — LSB, else spread-spectrum, else metadata, else C2PA — so nothing
  that read the result needs to change. What is new is `res.layers`: one entry per layer, never folded
  into a single number, with a status of `found` (a real watermark ID), `text` (a payload that isn't an
  ID), `none` (checked, nothing readable), `inconclusive` (the resize search ran out of time — not a
  confirmed absence) or `skipped` (not checked, with the reason: a `--mode`, no key to try, C2PA support
  not installed). On a reposted copy that is what tells you *which layers survived*: typically the
  metadata is gone and the spread-spectrum mark is what's left. Add `--layers` (Python: `layers=True`;
  HTTP: `{"layers": true}`) to print it; the data is in the Python/JSON result either way. **If two
  layers read two different IDs** a warning is printed regardless of `--layers` — a file stamped once
  carries the same ID everywhere, so a disagreement means it was stamped again, or one layer misread — and
  the headline is still just the first match in the order above, not necessarily the original stamp.
  One deliberate shortcut: when the LSB layer already read a real ID, `--rescale`'s resize search is not
  run for the spread-spectrum layer — a resize destroys an LSB mark, so that search could only burn its
  budget on a "not found"; the layer's note says so. The flip search still runs (a top-bottom flip does
  *not* break an LSB mark). Checking every layer costs real time when LSB reads first: a 6-megapixel
  image that used to return in 0.4 s now takes about 4 s (a 640×480 one, 0.03 s → 0.2 s) — the same as
  an image with no mark at all. Measurements and what was not measured:
  `bench/results/EXTRACT-LAYERS-NOTES.md`.

```bash
python3 cli/watermark.py in.jpg out.jpg --text "@yourhandle" --id --invisible --invisible-mode spread --c2pa
#   Watermark ID: XT55-RS4D-Z0 (carried by: spread, metadata, c2pa); recorded in ~/.watermark-tool/ledger.sqlite3. ...

python3 cli/extract_invisible.py suspect.jpg           # no key: the ledger's keys are tried
#   Watermark ID: XT55-RS4D-Z0
#   (found in the spread-spectrum mark, with a key from the ledger)
#   Ledger: stamped 2026-09-20T21:03:10+00:00 from 'in.jpg' as 'out.jpg' (image).
#   Layers that carry this ID: spread, metadata, c2pa.
#   This file is NOT byte-identical to the recorded output: it was re-encoded or edited after stamping -
#   or it is a different image that carries a copied ID. The ID alone can't tell those apart.

python3 cli/extract_invisible.py reposted.jpg --layers   # a re-upload: metadata stripped, pixels recompressed
#   Watermark ID: XT55-RS4D-Z0
#   (found in the spread-spectrum mark, with a key from the ledger)
#   ...
#   This file, layer by layer (each checked on its own - a hit in one says nothing about the others):
#     LSB mark:             nothing readable
#     spread-spectrum mark: found XT55-RS4D-Z0 (1 of 1 ledger key(s) tried)
#     image metadata:       nothing readable
#     C2PA manifest:        nothing readable

python3 cli/wm_ledger.py show XT55-RS4D-Z0             # everything on record (--secrets adds the key + settings)
python3 cli/wm_ledger.py find suspect.jpg              # is this exact file one we stamped? (by SHA-256)
python3 cli/wm_ledger.py list                          # newest first
```

```python
r = wmapp.watermark("in.jpg", "out.jpg", text="@me", stamp_id=True, invisible=True, invisible_mode="spread")
r.wm_id, r.id_layers                       # "XT55RS4DZ0", ["spread", "metadata"]
res = wmapp.extract("suspect.jpg")         # no key; res.wm_id, res.record, res.exact_match, res.scheme
[(l.layer, l.status) for l in res.layers]  # [("lsb","none"), ("spread","found"), ("metadata","none"), ("c2pa","none")]
```

**Batches** get one ID per file (they share one `--ss-key`, as before, so the ledger lists it once).
`--id-value` names one specific ID and so can't be used with `--batch`; an ID already in the ledger is
refused. **Video** can now also carry the ID in a hidden mark, same as images: add `--invisible
--invisible-mode spread` (see [Video support](#video-support) for what's different about it there
- a higher default `--ss-alpha`, per-segment key rotation, `extract_invisible.py` scanning every
frame). Without that, a video's ID lives only in the visible grid (`--grid-id`) and the C2PA
manifest (`--c2pa`); `--id` on a video with none of the three prints a warning, because then the ID
is recorded in the ledger but written into nothing the file carries.

**What an ID is not.**

- **Not a proof.** Anyone who can read an ID can copy it into a different file — trivially so from
  the metadata or a visible `--grid-id`. The ledger's output hash lets `extract` say "byte-identical to
  what I stamped", but for a re-encoded copy it can't tell "my image, recompressed" from "someone else's
  image carrying my ID". (Storing a perceptual hash in the ledger to compare content is not implemented.)
  The README's threat model above applies unchanged.
- **Not a secret.** The `--ss-key` is still a reproducibility value, not access control: roughly one
  wrong key in four still decodes a text payload, and a bit more often an ID (about 3 in 10 on
  synthetic images, see the next bullet; with `--resync` on, which retries the decode on a few shifted
  block grids, far more often — see "Crop resync"). The ID's check characters are what let `extract` accept a
  decode from a key that isn't the recorded one, and reject the junk other wrong keys produce.
- **Not a robustness upgrade over a short handle.** A shorter payload means more repeats per image, and
  that matters only for *sentence-length* payloads on *small* images: on synthetic 640×480 images a
  10-byte ID, an 11-byte handle and a 99-byte sentence were all detected 16/16 under every gated attack,
  but at 320×240 the 99-byte payload failed even unattacked (12/16) and under a 64 px top crop (0/16)
  while the ID and the handle stayed at 16/16 (`bench/results/p1-payload-length*`,
  `bench/results/P1-NOTES.md`; synthetic images, n=16 per cell). Nothing else about the tool's
  robustness claims changed with `--id`.
- **The hidden ID is read with a decoder that knows what an ID looks like.** An ID is always 112 bits
  on the wire, so `extract` doesn't hunt for the period (the text decoder needs one repeat's 32-bit
  header to survive bit-exact — with a noisy channel that fails long before the *sum* of the repeats
  does): it sums the repeats at each of the 112 possible starting phases and accepts only a canonical
  ID whose check characters verify (`wmcore.find_spread_spectrum_id`). Measured against the text
  decoder on the same attacked images (`bench/results/P2-NOTES.md`; synthetic images and 7 classic
  test photos — a direction, not a rate): **inside the documented range the gain is small** (768/768 vs
  765/768 on synthetic images, 112/112 both on the photos); it shows at the margin — JPEG q30, blur 1.5,
  and a JPEG q20 re-save through the full pipeline (0/6 → 4/6): 17 more images read, none fewer, in 1968
  paired synthetic trials. It never returned a wrong string (the text decoder returned 28, 24 of them on
  crops it isn't built for), and 0/400 synthetic and 0/28 photo mark-free images produced an ID. **The
  cost:** a wrong key reads the *right* ID more often — 29% of wrong keys on synthetic images (vs 17% for
  the text decoder; 24% vs 21% on the 7 photos, a difference that size can't resolve). It is the true ID,
  never a different one, but it is one more reason not to lean on `--ss-key` as a secret. `--rescale` reads
  with it too, at every candidate size (`bench/results/P2-NOTES.md` §6): on resized copies it read 8 more
  synthetic and 3 more photo trials than the text decoder and lost none — but those come from 3 of 24
  synthetic images and 2 of 7 photos, all of them cases where the text decoder had lost a mark that was
  there. The photos' ×0.5 misses are unchanged, and crop + resize was not measured. On never-stamped periodic
  patterns (checkerboards, stripes, grid lines; 10 hand-made images) the text decoder returned junk — accepted as
  a payload in 2 of 30 searches, both from the plain decode of a grid-lines image — while the ID reader never
  returned an ID (0 in 660 decode attempts; its own search accepted 0/30): a sign that the text decoder's false
  positives are real, not a rate. Wrong keys still read the true ID through the resize search about one time in
  five (23% vs 19% for the text decoder at 320×240, n=100, intervals overlapping).
- **Keep the ledger.** It is what turns an ID back into a file, a date and a key. Back it up; don't
  publish it (it holds keys). Lose it and the IDs are still readable, but nothing says what they mean.
  Local disk only — SQLite file locking is unreliable on network filesystems.
- **`--grid-id` makes the grid text longer** (wider tiles, a little more conspicuous). Measured effect on
  the hidden spread mark: none detectable — 16/16 in every configuration at 640×480 after identity, JPEG
  q60, JPEG q40 and a 0.75× resize with `--rescale`; at 1280×960 `gradnoise` the resize-recovery rate was
  3/10 for the legacy payload, for `--id` and for `--id --grid-id` alike (n=10, intervals wide). That 30%
  is the resize search's existing limit at this size — **and it comes from the visible grid itself, not
  from `--id`, the image content, or a general "large images are worse" rule.** The grid's tile
  `spacing`/`font_size` are fixed pixel values (not scaled to image dimensions), so which part of the
  tiled pattern lands at the image's top-left corner — the one spot the resize search's cheap scale-sweep
  statistic looks at — depends on image size. At 1280×960 that lands badly for `gradnoise`: turning the
  grid off (nothing else changed) takes that cell from 0/12 to 12/12; at 640×480 the grid never hurts
  (12/12 either way); a plain, unresized decode is unaffected by the grid at any size (12/12) — only the
  resize-search's corner-anchored statistic is fooled. `pink`'s own texture partly masks it (8-10/12
  either way, not the same total wipeout). Confirmed on one real photo too: a photographed checkerboard
  (`board.jpg`, strongly periodic content, same mechanism as the tiled grid) is the one real image in a
  small real-photo run that failed `resize_0.75 (+rescale)`. Not fixed — see
  `bench/results/P0-NOTES.md` §1 for the full account, the reproduction command
  (`bench/experiments/grid_corner_anchor.py`), and untried fix ideas (multi-point anchoring; scaling the
  grid to image size). Details in `bench/results/P1-NOTES.md`.
- **The web page's client-side mode has no ledger** (it is a static canvas approximation). In *Full
  pipeline* mode the server records into its own ledger (`serve.py --ledger`); a request can't choose the
  ledger file, and `/api/extract` returns the record without the key or settings.

## Known limitations

- **Tile overlap at default jitter** — the per-tile position jitter (an
  internal default of `create_watermark_layer` in `wmcore.py`, currently
  ±0.35 of `--spacing`) is not yet exposed as its own CLI flag. Combined
  with the default `--spacing 250`, tiles that land close together can
  overlap slightly, which looks a bit busy in some regions of the image.
  It's cosmetic, not a correctness bug — if it bothers you, increase
  `--spacing` or shrink `--font-size` so tiles have more room.

## Video support

Point the CLI at a video file (`.mp4`, `.mov`, `.m4v`, `.avi`, `.mkv`,
`.webm`) and it's detected automatically — same command, no separate flag.
The watermark grid is built once and stamped onto every frame; frame
compositing is parallelized across `--workers` processes so it makes real
use of multiple cores.

If `ffmpeg` is on your `PATH`, output is re-encoded to H.264 with the
**original audio track muxed back in**, in a single pass. Without `ffmpeg`,
it falls back to OpenCV's own writer — the video still gets watermarked,
but the output has **no audio** and uses a much less efficient codec. Get
`ffmpeg` from [ffmpeg.org](https://ffmpeg.org) (or `apt install ffmpeg` /
`brew install ffmpeg`) for real use.

`--copyright` is image-only and is ignored (with a note) for video input.
`--c2pa` (and, when combined with it, `--author` — see [C2PA (Content
Credentials)](#c2pa-content-credentials)) works for video too.

### Invisible mark (`--invisible --invisible-mode spread`)

`--invisible-mode lsb` never survives video re-encoding (H.264 or the
OpenCV fallback are both lossy) — asking for it on a video produces a
warning and no hidden mark, not a broken one. `--invisible-mode spread`
does work for video: the same `embed_spread_spectrum()` block-DCT mark used
for images is embedded into **every frame**, inside the same worker pool
that composites the visible grid, so it doesn't cost a separate pass.

A few things are different from the image version:

- **The default embedding strength (`--ss-alpha`) is higher for video**
  (20.0 vs. images' 12.0) — measured (`bench/results/P4-NOTES.md`) against
  real `ffmpeg`/H.264 re-encoding at CRF 23 (`libx264`'s own default
  quality, not an aggressive attack): at the image default, only ~65-70% of
  frames decoded afterward; raising it to 18-20 closed that to 100% in the
  same test.
- **The key rotates every `--jitter-refresh-seconds`** (the same cadence
  the visible grid's jitter already uses, not a separate flag) — the same
  anti-frame-averaging reasoning as the grid's own jitter, but on a fixed
  time interval, deliberately **not** tied to the grid's scene-cut-triggered
  early boundaries, so that a decoder can recompute which key a given frame
  used from nothing but that frame's position, the video's fps, and
  `--jitter-refresh-seconds` (public, unlike `--ss-key` itself) — see
  `video.py`'s module docstring for why. A frame within a few positions of
  a real scene cut may use a key one segment off from what a fixed-interval
  guess expects, and simply not decode — that's expected on an otherwise
  correctly-keyed video, not a sign something's wrong.
- **`extract_invisible.py` scans every frame of the video**, not just the
  first one, and reports the full breakdown (how many frames out of how
  many decoded this ID) rather than collapsing it to a plain found/not
  found — a still image gets one decode attempt; a video gets one *per
  frame*, so unlike a still image, a low per-frame hit rate does not mean
  the mark is gone: a real video has far more frames than the handful this
  was measured on, and only one of them has to decode.
- **A video reports its layers too**, but only two of the four can exist: the spread-spectrum mark
  (the frame scan above) and the C2PA manifest. LSB and file-format metadata come back as `skipped`, with
  the reason (neither survives re-encoding / video has no such layer). C2PA is now also read when the
  frames already gave an ID, so a disagreement between them would show.

A crop that removes columns from the left and/or right (common when a repost changes aspect ratio) used to
break the video mark completely; `extract` now re-aligns each frame it could not read the plain way
(`--resync`, on by default; the report says how many frames needed it). Measured on one synthetic 640×480
clip, 60 frames, re-encoded with x264 at CRF 23: both sides trimmed 80 px 60/60, a 9:16-style centre strip
57/60, an off-grid crop 51/60 (CRF 28 off-grid: 19/60; before this, all 0/60) —
`bench/results/CROP-NOTES.md` §6. On real footage (OpenCV's vtest.avi, 768×576, 60 frames, and the same scaled to 1920×1080, 24 frames) every crop tried read every frame at CRF 23 and 28 (§10); not measured on long video, native 1080p/4K footage, or codecs other than x264.

### Crop resync (`extract --resync`, on by default)

The spread-spectrum mark is laid out in raster order over the original's 8×8 block grid, so a copy
that was **cropped** did not decode: a left/right trim changes the row length, and a screenshot trimmed to
the picture almost never starts on a multiple of 8 pixels. `extract` now works out where the original grid was,
from the copy alone: it re-finds the block-grid offset (one 8×8 correlation filter gives all 64 offsets at
once) and tries every row-length/phase combination (112 × 112, a table lookup after folding the image once,
done with an FFT). A hit is still only a canonical, check-valid ID. It runs when the unmirrored plain decode found
nothing, is cheap (tens of ms per key), covers **mirrored** copies too (the plain mirrored decodes are not run
any more: the resync search reads at least as many, measured, and picks the key and the orientation by
confidence), and reports `(recovered after re-aligning ... looks cropped)`. With `--rescale` it also reads a copy
that was **cropped and resized** (a screenshot at another zoom, trimmed): it sweeps the scale on a central patch,
refines it, and re-aligns at the restored scale. Measured (`bench/results/CROP-NOTES.md`): left/right and off-grid
crops 0 → all read on synthetic images and on 11 real photos (JPEG down to q60 there, q30 on synthetic), down
to ~25% of the image area on real photos (~5% on large synthetic ones, and on 10 MP real photos); crop + resize 84 of 84 on
7 real photos (3 zoom levels, 4 tries), and on two real 10 MP photos at x0.75 and x0.5 (not at x0.33, where the mark itself is gone);
0 false IDs on 55 mark-free images. On real footage (OpenCV's vtest.avi, 768×576 and scaled to 1080p, x264 CRF 23-28)
every crop kind went from 0 to all frames read. `--resync-budget-seconds` (default 10) caps the search; running out
reports *inconclusive*, never a wrong ID. `--no-resync` turns it off (and brings back the old mirrored decodes).

What it does **not** do, and what it costs:
- **Crop + resize needs `--rescale`** (about 0.4 s per ledger key to sweep, plus about 1 s per key to verify on an image with no mark,
  so it is not on by default), works for an unmirrored copy with its aspect ratio kept. Ledger keys are swept in rounds of 6, newest
  first, each round verified before the next is swept, so a copy of a recent stamp is read in a few seconds whatever the ledger's
  length (a 200-key ledger: 4.8 s, 42 real-photo cases against a 60-key ledger, true key last, 8 s budget: 38 read, the 4 misses all one
  densely textured photo). A copy that cannot be read, or an old stamp on a long ledger of a hard photo, runs out of its share of the
  budget and is reported as inconclusive together with how many keys it got through (`--rescale-budget-seconds` raises it). **Below about x0.5 the mark itself does not survive**, whatever the decoder
  (checked by restoring the exact scale: x0.4 and under unreadable). Real X screenshots (a photo shown at 100/75/50% in a
  1600x1000 screen): every crop of the screenshot reads at all three scales (with `--rescale` below 100%), and the whole
  screen reads at 100% and 75%; a photo at 50% in the corner of an otherwise black screen, not cropped, does not
  (bench/results/CROP-NOTES.md §8c). A screenshot that also changed colours/gamma, has an overlay or a different aspect
  ratio has not been measured.
- **Some widths cannot survive a left/right trim**, whatever the decoder: widths that are a multiple of 56
  blocks (448, 896, 1344, 1792 px ...) must keep at least 448 or 896 px; all other widths are fine
  (`bench/results/CROP-NOTES.md` §4). Top/bottom trims are never affected. Very narrow or very small crops (under
  ~25 blocks across, or ~10% of a small photo) can still be lost on real photos.
- **A wrong key reads the real ID much more often** with it on (it retries on shifted grids): 18 → 54 of 60 trials
  in the measured sample. The ID read is still the one actually embedded, never a made-up one, but the key is
  now even less of a gate. "Not a secret" was already the documented position. With several keys in the ledger the
  report picks the key and the orientation by the strongest read, not the first (an earlier version of this labelled
  44% of its copies with the wrong mirroring when the ledger held 8 keys).
- **Time:** an image with no mark now costs about what it did before this feature (0.18 s at 640×480 and 2.1 s at
  6 megapixels with a 1-key ledger, against 0.20 s and 4.1 s before crop resync existed: the three mirrored DCT decodes are gone), a
  100-key ledger 0.3 s extra at 1280×960. Copies whose wrong keys read the ID partly cost ~30-55 ms per such key.
- Free-text payloads are not recovered this way, only IDs (the same reason mirrored copies are ID-only).

### Repost round trip (`bench/repost_roundtrip.py`) — the test that has to be run on real platforms

Every robustness number in this README was measured against a local stand-in (Pillow JPEG, ffmpeg/x264, a viewer's resize); none
was measured against a platform. YouTube, Facebook, Telegram, TikTok, X and Instagram each resize, re-encode and strip metadata in
their own way, so the honest status is: **not measured on any real platform.** The script does the bookkeeping so that the run is
cheap: `prepare` stamps your own photos/videos (ID + spread-spectrum mark + ledger, as in normal use) into an `upload/` folder and a
manifest; you post them, download the posted copies back into a folder per platform, and `check` reads every downloaded file and prints
one row per file — whether an ID was read and which upload it is, what each layer (LSB / spread-spectrum / metadata / C2PA) found on its
own, whether the copy looks cropped or resized, how long it took — plus the uploads that no downloaded copy matched. Use `--rescale`
on `check` for images that a platform may have resized, and `--c2pa` / `--jpeg` on `prepare` for the extra layers / a JPEG variant.
`tests/test_repost_roundtrip.py` runs the same code against a local resize + JPEG q75 stand-in, which checks the bookkeeping only.

## Batch mode & performance

`--batch` processes a whole folder — images and videos can be mixed in the
same folder. Images are processed **in parallel** across `--workers`
processes (default: your CPU count) with a live progress bar; each video is
still processed one at a time, but its own frames are parallelized
internally the same way. A failure on one file (corrupt image, unreadable
video) is reported and skipped rather than stopping the whole batch.

Every file in a batch gets its own jitter pattern (deterministically derived
from `--seed` + filename), so a fixed `--seed` still gives you reproducible
output without every file in the batch sharing an identical watermark
pattern.

## Web tool

Open `web/index.html` in a browser (or serve it as a static page, see
below). Drag in **one or more** images, tune text / opacity / density / font
size / angle / color, and:

- **Download this image** — the currently-selected image only.
- **Download all as .zip** — watermarks every loaded image with the same
  settings and bundles them into a single `.zip`, with a progress bar while
  it works. The zip is built entirely client-side (no library, no upload).

Switch between loaded images using the thumbnail strip above the preview;
each keeps its own original resolution on export. Video isn't supported in
the browser tool — use the CLI for that.

### Run it locally

```bash
# no build step — just open the file (fully client-side, nothing is uploaded)
open web/index.html        # macOS
xdg-open web/index.html    # Linux
```

### Full pipeline (optional server mode)

The page above draws a plain regular grid in your browser — it deliberately
has no jitter, blend modes, invisible marks, metadata or C2PA, because those
live in Python. To get them in the page too, run the bundled server and open
the page *from it*:

```bash
python3 cli/serve.py            # http://127.0.0.1:8765/  — the page + the API, this machine only
```

The page detects the server and shows a **Full pipeline (server)** switch,
off by default. Turn it on and the download buttons render on the server
with the *same code as the CLI and the Python API* (identical pixels — see
`tests/test_frontends_parity.py`); the extra controls (blend, ink, invisible
mark, key, author, adaptive placement, C2PA…) are built from the server's
`/api/schema`, so an option added in `cli/wmapp/options.py` appears here
without touching the page. The page says plainly when this is on that images
are sent to the server (and not stored); switch it off and everything is
back in the browser. The preview stays the local approximation — it can't
show jitter or invisible marks — so use **Render preview on server** to see
the real result, or just download.

The server is standard-library only and meant for your own machine. It binds
`127.0.0.1`; requests can't name server-side paths (font, certificate,
preview output, ledger — `--id` records into the server's own ledger,
`serve.py --ledger PATH`, and `/api/extract` returns the record without the
key or settings); upload size, pixel count, grid density and text length are
capped (`wmapp.SERVER_LIMITS`); the `Host` header must be a loopback name
(DNS-rebinding); and there is no CORS unless you pass `--cors-origin ORIGIN`
for one specific origin. Images only — video stays a CLI feature.

### Host it (e.g. GitHub Pages)

Since it's a static, single HTML file, you can publish it directly with GitHub
Pages: **Settings → Pages → Deploy from a branch**, and point it at the `web/`
folder (or copy `index.html` to the repo root / a `docs/` folder, depending on
how you want Pages configured).

## CLI tool

```bash
pip install -r requirements.txt
# or: pip install pillow numpy opencv-python-headless tqdm --break-system-packages
# (video output with audio also needs an ffmpeg binary on PATH)
```

`piexif` (in `requirements.txt`) is optional: it's only needed for
`--author`/`--copyright` metadata on **JPEG** output. If it's missing,
those flags are silently skipped with a printed note — everything else,
including PNG metadata, works without it. `c2pa-python`/`cryptography`
(also in `requirements.txt`, also optional) are only needed for `--c2pa` —
see [C2PA (Content Credentials)](#c2pa-content-credentials) — but unlike
`piexif`, `--c2pa` fails fast with a clear error if they're missing rather
than silently skipping the signature.

If **no TrueType font is found** (see `--font-path` below): with
Pillow >= 10.1 the tool falls back to Pillow's built-in scalable font, so
`--font-size` still works, just with a different typeface than DejaVu/Arial.
On older Pillow, the fallback is a tiny fixed-size bitmap font that ignores
`--font-size` (a warning is printed). Either way, pass `--font-path
/path/to/some.ttf` for full control (or `brew install --cask
font-dejavu-sans` / your distro's `fonts-dejavu` package to match the
Linux default look).

```bash
# single image
python3 cli/watermark.py input.jpg output.jpg --text "@yourhandle"

# single video (audio kept automatically if ffmpeg is installed)
python3 cli/watermark.py input.mp4 output.mp4 --text "@yourhandle"

# whole folder (images and videos mixed), parallel, reproducible jitter
python3 cli/watermark.py input_folder/ output_folder/ --text "@yourhandle" \
    --batch --seed 42 --workers 8

# texture-aware blend + invisible backup + provenance metadata (images)
python3 cli/watermark.py input.jpg output.png --text "@yourhandle" \
    --blend multiply --invisible --author "Jane Doe" --copyright "(c) 2026 Jane Doe"

# invisible mark that survives JPEG recompression (spread-spectrum)
python3 cli/watermark.py input.jpg output.jpg --text "@yourhandle" \
    --invisible --invisible-mode spread --ss-key 20260918

# content-adaptive placement: denser + darker over faces/main subject
python3 cli/watermark.py input.jpg output.png --text "@yourhandle" \
    --blend multiply --adaptive --adaptive-strength 0.8 --adaptive-preview mask.png

# C2PA (Content Credentials) - signs the file, images AND video; see the
# dedicated section above for the self-signed-by-default caveat
python3 cli/watermark.py input.jpg output.jpg --text "@yourhandle" --c2pa
python3 cli/c2pa_verify.py output.jpg

# read back the invisible mark later (scans the whole image, not just one corner)
python3 cli/extract_invisible.py output.png
python3 cli/extract_invisible.py output.jpg --mode spread --ss-key 20260918
# image was resized after stamping (slower; searches 0.5x-2x by default)
python3 cli/extract_invisible.py resized.jpg --mode spread --ss-key 20260918 --rescale
# --max-length caps how long a payload it will consider plausible while
# scanning (default 10000 bytes); raise it only if you embedded something
# longer than that with --invisible-text

# a short ID per output, recorded in a ledger (see "Watermark ID & ledger")
python3 cli/watermark.py input.jpg output.jpg --text "@yourhandle" --id \
    --invisible --invisible-mode spread
python3 cli/extract_invisible.py suspect.jpg        # no key needed: the ledger has it
python3 cli/wm_ledger.py show K3M9-QXR2-T4
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--text` | *(required)* | Watermark text, e.g. `@yourhandle` |
| `--opacity` | `90` | Base opacity, 0–255 |
| `--font-size` | `36` | Font size in px |
| `--angle` | `30` | Base rotation angle in degrees |
| `--spacing` | `250` | Spacing between watermark tiles, in px |
| `--font-path` | *(auto)* | Path to a custom `.ttf` font. Auto-detect tries DejaVu Sans Bold (Linux), Arial Bold (macOS), then Arial Bold (Windows), in that order; see `DEFAULT_FONT_PATHS` in `cli/wmcore.py` |
| `--batch` | off | Treat input/output as folders (images + video, mixed) |
| `--workers` | CPU count | Parallel workers: files-in-parallel for `--batch` images, frames-in-parallel for a video |
| `--seed` | random | Seed for per-tile jitter (reuse to reproduce the same pattern) |
| `--blend` | `alpha` | `alpha`, `multiply`, or `overlay` — how the mark mixes with the base pixels |
| `--ink` | `0.32` | Mark darkness for `multiply`/`overlay`, 0–1 |
| `--invisible` | off | Also embed a hidden backup mark; see `--invisible-mode` — video only supports `spread` |
| `--invisible-mode` | `lsb` | `lsb` (pixel LSB, fast, no key, PNG-only, images only) or `spread` (block-DCT spread-spectrum, survives JPEG/H.264 recompression + block-aligned crop, needs `--ss-key`, works for video too) |
| `--ss-key` | *(random, printed)* | Key for `--invisible-mode spread`. Reproducibility value like `--seed`, **not** a secrecy mechanism — see above |
| `--ss-alpha` | `12.0` images / `20.0` video | Embedding strength for `--invisible-mode spread`. Higher survives harder recompression/noisier images at the cost of a small but real visual imprint |
| `--invisible-text` | *(same as `--text`)* | Text for the hidden mark (ignored with `--id`, which embeds the ID) |
| `--id` | off | Stamp a short random watermark ID into every layer that can carry it and record it in the ledger; see [Watermark ID & ledger](#watermark-id--ledger---id) |
| `--id-value` | *(random)* | Use this specific ID (10 characters incl. check characters); must not already be in the ledger; not with `--batch` |
| `--grid-id` | off | With `--id`: also print the ID in the visible grid text (works for video). Makes the grid text longer |
| `--ledger` | `$WATERMARK_TOOL_LEDGER`, else `~/.watermark-tool/ledger.sqlite3` | Ledger file for `--id` (created `0600`; it stores spread-spectrum keys) |
| `--author` | — | Author name, embedded in EXIF/PNG metadata (images only) |
| `--copyright` | — | Copyright string, embedded in EXIF/PNG metadata (images only) |
| `--adaptive` | off | Detect faces/high-detail "main subject" regions and increase density there (+ darkness for multiply/overlay) |
| `--adaptive-strength` | `0.6` | 0-1, how strongly to boost density/darkness in detected regions |
| `--adaptive-detail-weight` | `0.5` | 0-1, weight of the general detail signal relative to face detection |
| `--face-cascade` | *(bundled)* | Path to a custom Haar cascade XML for face detection |
| `--adaptive-infill-divisor` | `2.0` | How much denser the extra grid is inside detected regions (spacing ÷ this) |
| `--adaptive-preview` | — | Images, single-file only: save the detected sensitivity mask as a PNG, for tuning |
| `--adaptive-track` | on | Video only: keep the sensitivity mask tracking the subject/camera across the clip (optical flow between periodic re-detections) instead of freezing it at frame 0. `--no-adaptive-track` for the old behavior |
| `--adaptive-redetect-seconds` | `1.0` | Video only, with `--adaptive-track`: how often to re-run full detection between flow-tracked frames |
| `--scene-cut-detect` | on | Video only: force an early jitter (and, with `--adaptive-track`, mask) refresh right at a detected hard scene cut. `--no-scene-cut-detect` to disable |
| `--scene-cut-threshold` | `0.12` | Video only: 0-1 grayscale-change threshold (on a downscaled frame) above which two consecutive frames count as a cut |
| `--no-shared-memory` | off | Video only: disable the shared-memory frame transport, fall back to a slower pickle-per-frame path (useful if shared memory isn't available in your environment, e.g. some sandboxes without `/dev/shm`) |
| `--jitter-refresh-seconds` | `4.0` | Video only: regenerate the jitter pattern every N seconds of output (rebuilds the worker pool at each boundary — see note below). `0` disables this and uses one static pattern for the whole clip — see the per-tile-jitter section above for why that's weaker |
| `--c2pa` | off | Sign the output with a C2PA (Content Credentials) manifest — images **and** video; see [C2PA (Content Credentials)](#c2pa-content-credentials) |
| `--c2pa-cert` | *(auto-generated dev cert)* | PEM file with your signing certificate (+ chain) for `--c2pa`. Must be given with `--c2pa-key` |
| `--c2pa-key` | *(auto-generated dev cert)* | Private key PEM matching `--c2pa-cert` |
| `--c2pa-alg` | `es256` | Signing algorithm for `--c2pa`; only relevant with your own `--c2pa-cert`/`--c2pa-key` — the auto-generated dev cert is ES256-only |
| `--c2pa-title` | *(output filename)* | Title recorded in the C2PA manifest |

## Architecture & Python API

The CLI, the web page's full-pipeline mode and Python callers all sit on one
application layer, `cli/wmapp/`, so a feature or a default exists in exactly one place:

```
   wmcore   image primitives: grid, adaptive placement, blend, LSB / spread-spectrum marks,
      ▲     the resize-recovery search, metadata
   wmapp    what the tool DOES: one option schema, validated requests, the watermark pipeline,
      ▲     batch fan-out, read-back, structured results and notes, resource limits
      │
   ┌──┴───────────────────────┬──────────────────────────────┬────────────────────────┐
   watermark.py               serve.py (stdlib HTTP)         your Python code
   extract_invisible.py       GET /api/schema                import wmapp
   c2pa_verify.py             POST /api/watermark|extract|verify
   wm_ledger.py
   argv → spec → call → print JSON/bytes → spec → call → JSON/bytes
```

A front-end parses its own input format into a `WatermarkSpec` / `ExtractSpec`,
calls one function, and renders the result its own way — none re-implements
behaviour, so they can't drift. `tests/test_frontends_parity.py` proves it:
the same job through the CLI (a real subprocess), the Python API and HTTP
gives identical pixels, and the hidden mark reads back identically through each.

```python
import sys; sys.path.insert(0, "cli")          # not pip-installable (yet): cli/ is the import root
import wmapp

r = wmapp.watermark("in.jpg", "out.png", text="@me", blend="multiply",
                    invisible=True, invisible_mode="spread")     # spread mark, key generated for you
r.ss_key, r.c2pa, r.seconds                      # the key to read it back, "off"/"signed"/"failed", timing
[n.message for n in r.notes]                     # everything the CLI would have printed - a library doesn't print
wmapp.extract("out.png", mode="spread", ss_key=r.ss_key).text   # "@me"
wmapp.extract("small.png", mode="spread", ss_key=r.ss_key, rescale=True)   # after a resize: search for the scale (see --rescale)

r = wmapp.watermark("in.jpg", "out.jpg", text="@me", stamp_id=True, invisible=True, invisible_mode="spread")
r.wm_id, r.id_layers                             # a short ID, and which layers carry it (see "Watermark ID & ledger")
wmapp.extract("suspect.jpg").record              # no key needed: the ledger's keys are tried; the ID's record comes back

wmapp.watermark_batch("photos/", "marked/", wmapp.WatermarkSpec(text="@me", seed=7), workers=4)
spec = wmapp.WatermarkSpec.from_mapping({"text": "@me", "opacity": 120})   # from JSON / a config file
wmapp.schema()                                   # every option: type, default, range, help, group
```

`watermark_batch` runs worker processes: on macOS/Windows (the `spawn` start method) call it from
under `if __name__ == "__main__":`, as the CLI scripts do.

Errors are `wmapp.SpecError` (a `ValueError`: bad value, unknown option, contradictory flags),
`UnavailableError` (C2PA packages missing) and `LimitError` (a configured `Limits` exceeded —
the CLI and Python API default to none; the HTTP server applies `SERVER_LIMITS`).

**Adding or changing an option is one edit** — an entry in `cli/wmapp/options.py`. From that
list are generated the CLI flag (and its `--help` text), the `WatermarkSpec` field with its
default and range check, and the entry in `/api/schema` that the web page renders. Options
that name a path on the machine running the code (`local=True`: font, certificate, preview
output) exist for the CLI/Python API and are refused over HTTP.

What this does *not* unify: the static, client-side web mode is still its own small canvas
implementation (different font, no jitter) and is an approximation of the real output, not a
copy of it; and video is CLI/Python only.

## Project structure

```
.
├── cli/
│   ├── wmcore/                 # Image primitives: grid, adaptive placement, blend, invisible marks (LSB, spread-spectrum, resize-recovery search, mirrored-copy search in orient.py) + their read-back, metadata, watermark-ID format (wmid.py)
│   ├── wmapp/                  # Application layer shared by every front-end (see "Architecture")
│   │   ├── options.py          #   THE option list -> CLI flags, WatermarkSpec/ExtractSpec, /api/schema
│   │   ├── pipeline.py         #   prepare(), watermark(), watermark_file(), watermark_batch()
│   │   ├── extraction.py       #   extract() (ID- and ledger-aware), verify(), capabilities()
│   │   ├── ledger.py           #   the SQLite ledger behind --id: ID -> file, hashes, key, layers, settings
│   │   └── limits.py, events.py, errors.py
│   ├── watermark.py            # CLI shell over wmapp: images, batch, video
│   ├── extract_invisible.py    # CLI shell: reads back either invisible-mark scheme (lsb or spread); with --id marks, the ID + its ledger record
│   ├── wm_ledger.py            # CLI shell: look up / list / find IDs in the ledger
│   ├── c2pa_verify.py          # CLI shell: reads back a --c2pa manifest
│   ├── serve.py                # HTTP shell (stdlib): serves web/ and /api/{schema,watermark,extract,verify}
│   ├── video.py                # Video pipeline: parallel frame workers, ffmpeg mux/encode, per-frame invisible mark
│   └── c2pa_provenance.py      # C2PA (Content Credentials): dev-cert gen/reuse, sign_file(), read_manifest()
├── tests/                      # pytest: primitives, the application layer, CLI shells, HTTP adapter, cross-front-end parity, bench/
│   └── js/                     # Node tests for the web page's server mode (client module; page boot in jsdom)
├── bench/                      # measurement framework: detection rate / BER / FPR / runtime under attack (see bench/README.md)
├── web/
│   └── index.html              # Browser tool: multi-image batch, zip export; client-side by default, full pipeline via serve.py
├── requirements.txt
├── requirements-dev.txt        # requirements.txt + pytest
└── README.md
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/
```

Covers the blend-mode math (`wmcore.blend_arrays`, all three modes), the
content-adaptive placement math (`wmcore.build_ink_map`, sensitivity-map
shape/range guarantees), both invisible-mark embed/extract round trips
(`lsb` and `spread`), and C2PA sign/read round trips including the
dev-cert generate-then-reuse behavior and the self-signed
`signingCredential.untrusted` expectation - these are the places most
likely to silently regress since none has an obvious "it crashed" failure
mode (a wrong blend, a broken payload, or a manifest missing a field all
just *look* fine at a glance). The C2PA tests are skipped automatically if
`c2pa-python`/`cryptography` aren't installed.

The application layer is tested three ways: directly through the Python API
(`test_wmapp.py`: schema consistency, validation, notes, limits — including that
the tile-count estimate equals what `wmcore` really draws — batch, video,
extract, backward compatibility), through each shell (`test_cli_shell.py` for
argparse/exit codes, `test_serve.py` for the HTTP API and what it refuses), and
across them (`test_frontends_parity.py`: identical pixels and identical
read-back from CLI, Python and HTTP — for `--id` stamps too). The ID format is
`test_wmid.py`, the ledger `test_ledger.py` (including several processes writing
at once), and `--id` end to end — every layer, batch, video, `extract` with and
without a key, a resized or recompressed copy, a junk LSB decode — `test_id_ledger.py`; mirrored
copies (every orientation, an odd image size, a JPEG re-save, the ordering rule that stops junk text
beating a mirrored ID, and `extract`'s wording) `test_orient.py`; the per-layer report of `extract` (every layer's own outcome, why a layer was not
checked, the disagreement warning, the plain-decode shortcut, video) `test_extract_layers.py`; crop re-alignment (grid filter == DCT correlations, off-grid / left-right / mirrored
crops, JPEG, mark-free images, the width limit, `extract`, video, the mirroring label with several ledger keys, the pre-screen, crop + resize) `test_resync.py`.
`conftest.py` points the default ledger at a per-test path, so no test touches a
real `~/.watermark-tool/ledger.sqlite3`. `test_web_client.py` runs the JavaScript
tests in `tests/js/` against a real server; it needs Node ≥ 20 and is skipped
without it (run `npm install` in `tests/js/` to also enable the jsdom test that
boots the whole page).

## Benchmarks

`pytest` says whether a case passes; it can't say *how often* a mark
survives, how much signal is left when it doesn't, how often a decoder
reports a payload that isn't there, or how long the rescale search takes.
`bench/` measures those:

```bash
python3 bench/run_benchmark.py --suite quick       # ~1-2 min, one core
python3 bench/run_benchmark.py --suite standard    # the real measurement
python3 bench/run_benchmark.py --suite standard --images ~/photos   # on real photos
python3 bench/run_benchmark.py --suite quick --baseline bench/baselines/quick.json  # CI gate
python3 bench/run_benchmark.py --suite quick --decoders ss,ss_id --payloads wmid    # text vs ID-aware decoder, same images
```

Each run writes `report.md` (detection rate with 95% intervals, raw and
combined BER against oracle alignment, false-positive rate on unmarked images
and on wrong keys, embed/decode/rescale runtime p50/p95) plus raw
per-trial records. See [bench/README.md](bench/README.md) for definitions,
how to read a report, and how to extend it.

## License

MIT — see [LICENSE](LICENSE).
