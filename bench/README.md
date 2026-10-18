# bench/ — measurement framework for the invisible marks

`pytest` (see `tests/`) answers *"does this specific case still work?"* — one
image, one seed, pass or fail. It cannot tell you that a mark survives JPEG 50
in 98% of images but only 71% at JPEG 30, that the LSB extractor reports text
on marks that were never there, or that a change made `--rescale` twice as
slow. This directory measures those things.

```
python3 bench/run_benchmark.py --suite quick        # ~1-2 min, one core: sanity / CI
python3 bench/run_benchmark.py --suite standard     # tens of minutes: the real measurement
python3 bench/run_benchmark.py --suite full         # payload x strength x size sweep
python3 bench/run_benchmark.py --list-attacks
```

Nothing new to install: it uses `numpy`, `Pillow`, `opencv` (already in
`requirements.txt`) and `tqdm` for the progress bar if present.

## What is measured

| Metric | Definition | Why it is separate |
|---|---|---|
| **Detection rate** | exact payload recovered ÷ marked-and-attacked images, with a 95% Wilson interval | the headline number |
| **Misdecode rate** | a *different* text returned ÷ same images | a wrong answer is worse than "not found"; a plain miss and a confident-wrong are counted apart |
| **BER (raw)** | hard-decision error of every block (spread) or pixel (LSB) against the true tiled bit sequence, before combining repeats; 0.5 = noise | channel quality |
| **BER (combined)** | per-bit error after summing every repeat at the true period/phase — what `spread.py`'s "processing gain" buys | shows a signal is still there even when raw BER is double-digit |
| **FPR — unmarked** | any text returned ÷ images that were never stamped (also after JPEG / resize), exact Clopper–Pearson interval | a decoder that says "found: x" on nothing is wrong whatever x is |
| **FPR — wrong key** | any text returned ÷ marked images decoded with a random other key; `payload recovered` counts exact leaks | measures the README's "~1 in 4 wrong keys still decode" (text payloads; `ss_id` reads IDs and leaks more — `results/P2-NOTES.md`) |
| **Runtime** | wall-clock embed and decode, p50 / p95, per image and per megapixel; rescale search separately, with `budget hit` counts | |
| **PSNR** | embedding distortion (dB) | a guard rail: raising `--ss-alpha` makes detection look better monotonically, so every sweep needs its cost next to it |

### BER is measured with oracle alignment — deliberately

The benchmark knows what attack it applied, so it can map every surviving
block back to its position in the original tiling (a block-aligned crop
shifts the raster index; a resize is undone exactly) and compute BER with
the phase *given* rather than *searched for*. That makes BER a statement
about the signal, independent of the decoder, and lets the report tell two
failures apart that detection rate alone conflates:

* **signal intact – decoder loses sync**: combined BER ≈ 0 but detection
  fails. The mark is there; the decoder could not find its way to it
  (e.g. plain `ss` on a resized image, or the header-bootstrap in
  `find_spread_spectrum_payload` needing one error-free 32-bit window when
  raw BER is high).
* **signal destroyed**: combined BER near 0.5. Nothing to find.

Geometries with no oracle (unaligned crop, rotation) report BER as `n/a`.

## Corpus

Synthetic by default, so a run needs no data and is exactly reproducible
from `--seed`:

* `pink` — 1/f-spectrum texture + smooth colour field + hard-edged shapes +
  mild sensor noise (much closer to a photo's DCT statistics than the unit
  tests' gradient + white noise, and harder).
* `gradnoise` — the unit tests' image, for continuity with them.

**Synthetic images are a stand-in.** For any number you intend to publish or
tune against, point `--images DIR` at real photos (`--n-images` caps how many;
long side is capped by `--max-side`, default 1600). Every report states which
corpus it was measured on.

## Attacks and scope

`--list-attacks` prints all of them: identity; JPEG q 95/85/75/60/50/30, WebP
q80 (if Pillow has WebP), and AVIF q60 (if Pillow has AVIF); uniform resize
0.5–1.5× (plain, and followed by JPEG 85); block-aligned crops, unaligned
crops at a few pixel offsets; Gaussian noise σ 2/5/10; Gaussian blur
0.5/1.0/1.5; brightness / contrast / gamma; rotation at 1/2/3/5°; a
horizontal flip; two perspective-warp strengths; and two simulated
repost **pipelines** (resize + JPEG + crop, chained in one pass) standing in
for a real platform's combined resize/recompress/crop, not one isolated
transform at a time — see the `pipeline_*` rows in a report and
`bench/results/p0-new-attacks/report.md` for the run these were added and
first measured in (2026-09-20), plus `bench/results/p0-new-attacks-n12/`
and `bench/results/P0-NOTES.md`'s 2026-09-21 update for the larger follow-up
those numbers below cite: AVIF q60 and WebP q80 decoded cleanly at n=4/kind
in the pilot and again at n=12/kind (24/24 combined, 0 misdecode) in the
follow-up — same shape as the already-gated `jpeg_*` rows on this synthetic
corpus, so `avif_60` is now gated too (still worth a real-photo run before
treating the number as final; see *Corpus* above). A horizontal flip
produced a **misdecode** (a confident wrong string, not just a miss) in
1/8 trials at pilot scale and 1/24 at the n=12 follow-up — consistent
direction, still too thin to quote a rate. Both pipeline attacks failed
0/3 against `ss_rescale` even at the full default 20s budget (confirmed not
budget-starved — runtime stayed under 10s), which is a measured data point
behind the "a resize combined with a crop is not searched for" limit
already noted above, not a new one.

A dagger (†) in a report marks a row that is **measured but not gated**:
either the README documents it as unsupported (unaligned crop, rotation,
flip, perspective, the repost pipelines, 0.5× resize, a left-side crop that
changes the block-column count), or it is outside that decoder's design
scope (plain `ss` on resizes — that is what `--rescale` is for; `lsb` on
anything lossy). Those numbers show *why* a feature exists; they never fail
a regression check.

## Outputs

`bench/results/<timestamp>/` (git-ignored; override with `--out`):

| File | Contents |
|---|---|
| `records.jsonl` | one line per trial: outcome, BERs, confidence, factor, repeats, `decode_s`, `embed_s`, PSNR, image id/size, alpha, payload id… |
| `summary.json` | per-group aggregates (the thing baselines are made from) + runtime table + run metadata |
| `summary.csv` | the same rows flat, for a spreadsheet / plotting |
| `report.md` | headline, per-decoder tables, FPR table, runtime table, α-sweep (when `--alphas` has >1 value) |
| `meta.json` | seed, config, Python/numpy/OpenCV/Pillow versions, CPU count, `jobs`, and a **fingerprint of `cli/wmcore/*.py`** so a result is always tied to the code that made it |

`--report-only DIR` re-renders `summary.*` and `report.md` from an existing
`records.jsonl` without re-running anything.

## Reading the numbers

* **Look at the interval, not the point estimate.** 3/3 is `[44%–100%]`.
  `quick` is for catching breakage; use `standard`+ for claims.
* **FPR needs many images.** With n negatives the smallest upper bound you can
  claim is ≈ 3/n (0 hits in 100 → ≤ 3.6%; in 300 → ≤ 1.2%). The report prints the
  *upper* bound because that is the honest figure. Raise `--n-neg` (unmarked
  decodes are cheap; only the rescale search is slow — `--n-neg-rescale`).
* **`confidence` (rescale decoder)** is the tool's own heuristic index, not a
  probability. The benchmark records it per trial so it *can* be calibrated
  once a real corpus exists; it does not pretend to.
* **`misdecode` > 0 on an attack row** is the number to worry about first.

## Regression gate (CI)

```
# once, on a known-good commit and a fixed environment:
python3 bench/run_benchmark.py --suite quick --save-baseline bench/baselines/quick.json

# in CI: exit code 1 if anything got worse
python3 bench/run_benchmark.py --suite quick --baseline bench/baselines/quick.json
```

Fails on: detection-rate drop > `--det-tol` (default 5 points) on gated rows,
combined-BER rise > 0.02, FPR rise > `--fpr-tol` (default 2 points). Because
the corpus and every attack are seeded, an unchanged codebase on the same
library versions reproduces the baseline exactly — any difference is a
behaviour change, not sampling noise. **Regenerate the baseline when you bump
Pillow/OpenCV/numpy** (the JPEG encoder and resamplers are part of the
measurement).

Runtime is only compared with `--gate-runtime FACTOR` (e.g. `2.0` = fail if
decode p50 doubles), and only meaningful against a baseline from the *same
machine*.

## Timing hygiene

* Use `--jobs 1` for anything you will quote. `--jobs N` parallelises units
  across processes — great for detection/BER/FPR sweeps, but workers compete for
  cores and inflate every runtime figure (the report says so when `jobs > 1`).
* One warm-up embed/decode/warp runs before anything is timed (first-call
  einsum/OpenCV setup would otherwise land in the first record).
* numpy/OpenCV may use several threads internally; `meta.json` records the CPU
  count. For cross-machine comparisons pin threads
  (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`).
* Timings are wall-clock, the same clock the rescale budget (`max_seconds`) uses.

## Extending

* **New attack**: write `fn(rgb, rng) -> (rgb, Geometry)` in `bench_attacks.py`
  and `_register(name, family, fn, gate=...)`. Return `Geometry("resize", scale=…)`
  or `("crop_blocks", crop_blocks=(rows, cols))` if BER can be aligned, else
  `Geometry("none")`. Use only the passed `rng`, so runs stay reproducible.
* **New decoder/scheme**: add to `DECODERS`, `DECODER_SCHEME` (and `embed`,
  `oracle_soft` for a new scheme) in `bench_schemes.py`, and declare its
  `OUT_OF_SCOPE` families in `run_benchmark.py`.
* **New payload length**: add to `PAYLOADS`. Longer payloads mean fewer repeats
  per image and so less processing gain — `--payloads id,short,medium,long`
  (`id` is a 10-byte ID-*shaped* string whose check characters are **wrong**
  (`wmid.parse` rejects it): a length stand-in only). In principle; at 640×480 the
  measured effect on the gated attacks is nil, see `results/P1-NOTES.md`.
  `wmid` is a real, check-valid watermark ID, generated fresh per image from the
  seed — the payload any decoder that verifies IDs needs.
* **Payload-specific decoders**: `ss_id` (`wmcore.find_spread_spectrum_id`) reads
  only IDs, so `DECODER_PAYLOAD` runs it only on the `wmid` payload (it would just
  say "none" on free text). `ss_id_rescale` is the same reader inside the resize
  search; `ss_or_id_rescale` is "an ID if there is one, else text" (what `extract
  --ss-key --rescale` uses). Rescale decoders are slow (seconds per image, more on
  unmarked ones) and only run on the rescale-eligible images / attack families. Compare it with the text decoder on the very same
  attacked images with `--decoders ss,ss_id --payloads wmid`; both are scored
  against the same embedded ID, and unmarked-image / wrong-key trials run for both.
* **Mirror-aware decoders**: `ss_flip` (an ID in any of the 4 orientations, else the text decoder on the image as
  given - what `extract --ss-key` uses) and `ss_id_flip` (ID only), wmcore/orient.py. The attack family `flip`
  (`flip_vertical`, `flip_both`, `flip_horizontal_jpeg_75`, `flip_vertical_jpeg_50`, all with an oracle geometry, so
  BER says whether the signal survived) is out of scope for the plain decoders (`OUT_OF_SCOPE`) and in scope for these
  two - `ss_flip` only on the `wmid` payload (`IN_SCOPE_ONLY_FOR_PAYLOAD`: a mirrored text payload is not read, by
  design). `flip_horizontal_resize_0.75` / `flip_vertical_resize_1.25` (family `pipeline`, mirror + resize) are
  measured, not gated: no decoder here reads them. Run `--decoders ss,ss_id,ss_flip,ss_id_flip --payloads
  wmid,short,medium` to see plain and mirror-aware readers on the same attacked images.
* **One-off experiments** that need the real pipeline rather than the embed-only
  path live in `experiments/` (e.g. `id_rescale_hypotheses.py`: how many decode attempts the resize search makes on never-stamped images (also `--structured` for periodic patterns, `--id-search`); `id_rescale_wrongkey.py`: wrong keys on resized copies; `paired_decoders.py`: paired comparison of decoders from a run's `records.jsonl`; `flip_text_probe.py`: would the text decoder be safe on mirrored orientations (it is not: junk on periodic images)
  and what the mirror-aware decoder costs; `id_decoder_probe.py`: where the ID-aware decoder
  beats the text one and how many payload bits are still wrong when it misses;
  `grid_id_interference.py`: does `--grid-id`
  hurt the hidden mark?; `video_invisible_probe.py`: pre-implementation probe - embeds the same
  `embed_spread_spectrum()` used for images into every frame of a synthetic panned video, encodes with
  real ffmpeg/libx264 (CRF, fps, crop, letterbox), decodes per frame - `results/P4-NOTES.md` for numbers;
  `video_alpha_sweep.py`: follow-up, does raising `alpha` close the CRF-23 gap `video_invisible_probe.py`
  found). They reuse the corpus and the Wilson intervals from here
  and write a `report.md` under `results/` like a benchmark run does. (The two `video_*` probes are an
  exception - no `cli/video.py` code exists yet to call, so they drive `wmcore.embed_spread_spectrum()`
  and ffmpeg directly instead of the real pipeline.)
* Modules use flat top-level imports like `cli/` does, so worker processes
  under the `spawn` start method can pickle everything by reference.

## What this does not cover

The visible grid/jitter (no decoder to score), C2PA (a signature verifies or
it doesn't — `tests/test_c2pa.py` is the right tool), and video. Nor
regeneration/diffusion-purification or adversarial attacks (see the README's
threat model) — those need a model in the loop; the attack registry is where
they would go. And synthetic images are not photos: see *Corpus*.

**`--resume`** continues a killed run into the same `--out` with the same arguments: units whose records are already in `records.jsonl` are skipped and a torn last line is dropped. `wall_seconds` in `meta.json` then covers only the resumed part.
