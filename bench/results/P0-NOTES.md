# P0-bis (2026-09-20/21) — real images, and the gradnoise@1280x960 anomaly explained

Follows on from §4.1 of `HANDOFF-layered-repost-resistance-v17.md`. Two things, both **[đã xác minh]**:

## 1. Why `resize_0.75 (+rescale)` only recovered 30% of `gradnoise` at 1280x960 (P1-NOTES §2)

**Root cause: the visible grid layer, not the image kind or size as such.**
`wmapp.watermark()` always composites the tiled grid text (`create_watermark_layer`,
`wmcore/visible.py`) *before* embedding the spread-spectrum mark — including in the P1
experiment's "config A" baseline, which still carried a short grid text. `_build_grid_layer`
uses **fixed-pixel** `spacing` (default 250) and `font_size` (default 36) — neither scales with
image dimensions — and crops the tiled pattern from a `diagonal x diagonal` canvas *centered* on
the image (`left = (diagonal - width) // 2`). That crop offset depends on image size, so the exact
grid phase landing at the image's **top-left corner** is size-dependent. The rescale search's cheap
coarse-sweep statistic (`_ss_patch_bimodality`, `rescale.py`) is anchored *only* at that corner (for
cost reasons — see the module docstring), so a size/content combination that puts unfavorable grid
texture there can defeat the search even though the underlying mark is completely intact.

Confirmed with `bench/experiments/grid_corner_anchor.py` (n=12 seeds each, `--rescale-budget 10`):

| image | plain decode (no resize), grid ON | resize 0.75x + rescale, grid ON | resize 0.75x + rescale, grid OFF |
|---|---|---|---|
| gradnoise 640×480 | 12/12 | 12/12 | 12/12 |
| gradnoise 1280×960 | **12/12** | **0/12** | **12/12** |
| pink 1280×960 | 12/12 | 8/12 | 10/12 |

Reading this: the plain (unresized) decode is unaffected by the grid at any size (12/12) — the mark
itself is fine. The resize+rescale-search path is where it breaks, and only for this specific
size/content combination: gradnoise at 1280×960 goes from 12/12 to 0/12 purely by switching the grid
off (`opacity=0`, same layer, so nothing else changes). At 640×480 the grid never hurts. `pink`'s own
broadband texture partly — not fully — masks the interference (8-10/12 either way, a real but much
smaller effect than gradnoise's total wipeout).

This also shows up **outside synthetic images**: in the P0-bis real-photo run below, `board.jpg`
(a photographed camera-calibration checkerboard — i.e. real, strongly periodic content) is the one
real image that failed both `blur_1.0` on the plain `ss` decoder and `resize_0.75 (+rescale)`, while
every other real photo passed both. A checkerboard is a different kind of periodic structure than the
tiled grid, but the common thread — strong periodic/structured content interacting badly with a
statistic computed on a small, fixed corner patch — matches the mechanism above. This is evidence the
finding generalizes past "gradnoise is an unrealistic synthetic image," not proof; n=1 real image
showing the symptom isn't enough to say how often it would bite on real repost content.

**Not fixed, just diagnosed.** Options for later (not started, no bench numbers for any of them):
anchor the coarse sweep's cheap patch at more than one location instead of only the top-left corner;
scale grid `spacing`/`font_size` with image dimensions instead of using fixed pixel defaults; or just
document the limit. Given "measure before adding a layer," the anchor-diversification idea should be
bench'd cheaply (it only touches `_ss_patch_bimodality`'s call sites, not the embed format) before
picking one.

Reproduce: `python3 bench/experiments/grid_corner_anchor.py --n 12 --kind gradnoise --size 1280x960
--out bench/results/p0bis-grid-corner-anchor` (and `--size 640x480`, `--kind pink`). ~150-220s each on
1 CPU. Raw per-seed results in `bench/results/p0bis-grid-corner-anchor/*.json`.

## 2. First real-image run (`--images`), trimmed to fit a short session

7 real photos (`fruits.jpg`, `baboon.jpg`, `board.jpg`, `butterfly.jpg`, `aloeL.jpg`, `lena.jpg`,
`starry_night.jpg` — standard OpenCV `samples/data` test images, 356×493 to 1110×1282, not this
project's own corpus) — **[chưa xác minh] mở rộng: chỉ 7 ảnh, n=1-2 mỗi ô, không đủ để trích dẫn tỷ
lệ % làm số liệu chính thức, chỉ đủ để phát hiện điều bất thường như ở trên.** Full `standard` suite
is "tens of minutes" per its own docstring and gets killed by this sandbox's ~10 min background-process
limit, so this run trimmed `--attacks`/`--n-neg`/`--n-wrongkey-images`/`--rescale-budget` to fit in one
~280s pass:

```
python3 bench/run_benchmark.py --images <dir of 7 real photos> --n-images 7 \
  --attacks identity,jpeg_75,jpeg_50,webp_80,resize_0.75,resize_1.25,noise_5,blur_1.0,crop_unaligned_3,pipeline_resize_jpeg_crop \
  --rescale-images 3 --n-neg-rescale 2 --n-neg 7 --n-wrongkey-images 3 --n-wrong-keys 3 \
  --rescale-budget 10 --payloads short --out bench/results/p0-bis-real --no-progress
```

Headline (`bench/results/p0-bis-real/report.md`, full per-image tables there):
- `ss` (plain spread decode): 41/42 = 97.6% over gated attacks — matches synthetic-image expectations.
  The one miss is `board.jpg` on `blur_1.0` (see §1 above).
- `ss_rescale`: 6/9 = 66.7% — the one resize_0.75 miss is again `board.jpg` (checkerboard, §1's
  real-image analogue).
- **`lsb` false positives on unmarked real photos: 3/14 = 21.4%** (upper 95% bound 50.8%) — notably
  higher than what the synthetic-image pilots showed. n is far too small to call this a rate, but it's
  the first sign that LSB's false-positive behavior on real sensor noise might differ from synthetic
  noise and is worth a dedicated real-image FPR run before quoting LSB's FPR from synthetic images
  alone.
- `ss` wrong-key acceptance: 2/18 = 11.1% here (small n; README's "~1 in 4" from P1-NOTES §3 used 501
  keys on one image and remains the better-supported number — this run doesn't contradict it, just
  doesn't have the n to refine it).

Not done yet: a real, full-scale `standard`/`full` run (needs either a longer session, `--jobs`
parallelism on more CPUs, or chunking the suite into several sub-280s invocations and merging
`records.jsonl` — none implemented); a dedicated real-image LSB-FPR run with more images to see if
21.4% holds up; sourcing a larger, more varied real-photo corpus (these 7 are classic CV test images,
not repost-style content — no screenshots, no social-media recompression artifacts, no phone photos).



These two directories are **pilot-scale** measurements (n=3-4 images per
image kind, not the full `standard` suite's n=10) made while adding the new
attacks in `bench/bench_attacks.py` (see the handoff doc's P0 step). They're
kept here, un-gitignored-in-spirit, specifically so the citations to them in
the main README and `bench/README.md` point at something real. Re-run at
full `standard` scale before treating any of these as a stable, quotable
number — several are already structural/deterministic (won't change with
more samples: crop_unaligned_*, rotate_*, flip, perspective all fail because
the transform genuinely destroys the oracle's phase alignment, not because
of noise), but the AVIF/WebP "100% at n=4" results and the flip misdecode
rate are thin and should not be repeated as-is without a bigger n.

- **p0-new-attacks/** — the main pass: all new attacks (avif_60, the four
  crop_unaligned_* offsets, rotate_1/2/3/5deg, flip_horizontal,
  perspective_light/medium, both pipeline_* attacks) plus a few existing
  ones for context (identity, jpeg_75, resize_0.75, webp_80), on `ss` and
  `ss_rescale`, `--rescale-budget 6` (short, to keep the pilot fast).
  Command (see `meta.json` for the exact config):

  ```
  python3 bench/run_benchmark.py --suite standard \
    --kinds pink,gradnoise --sizes 640x480 --n-images 4 --rescale-images 4 \
    --decoders ss,ss_rescale --rescale-budget 6 \
    --attacks identity,jpeg_75,resize_0.75,webp_80,avif_60,crop_unaligned_1,crop_unaligned_3,crop_unaligned_7,crop_unaligned_15,rotate_1deg,rotate_2deg,rotate_3deg,rotate_5deg,flip_horizontal,perspective_light,perspective_medium,pipeline_resize_jpeg_crop,pipeline_heavy_repost \
    --n-neg 4 --n-neg-rescale 2 --n-wrongkey-images 4 --n-wrong-keys 4
  ```

- **p0-pipeline-fullbudget/** — a follow-up on just the two `pipeline_*`
  attacks, at the *real* default rescale budget (20s, not the 6s used
  above), to check whether the 0/4 pipeline result in the main pass was
  just budget starvation. It wasn't: runtime stayed at 6.5-9.5s (well
  under the 20s cap — the search actually finished, "budget hit" column
  empty) and detection was still 0/3. That's the evidence behind the
  stronger claim in the README ("confirmed not budget-starved").

## Headline (see each report.md for full tables + CIs)

- `avif_60`, `webp_80`: 100% detection, n=4/image-kind — promising, not
  gated yet (too small a sample; see bench/README.md's Attacks section).
- `crop_unaligned_{1,3,7,15}`, `rotate_{1,2,3,5}deg`, `perspective_{light,medium}`:
  0% detection, as the README already says should happen (no oracle
  geometry either — BER reads n/a, not "signal gone but decoder confused").
- `flip_horizontal`: 0% detection + **1 misdecode** (a confident wrong
  string) out of 8 trials on gradnoise. Worth a bigger run to get an actual
  rate before README quotes one.
- `pipeline_resize_jpeg_crop`, `pipeline_heavy_repost` (resize + JPEG +
  unaligned crop, chained): 0/4 then 0/3 with `ss_rescale`, confirmed not
  budget-starved at the full 20s default. This is the bench-measured
  version of the README's existing "a resize combined with a crop is not
  searched for" sentence — not a new limitation, just now backed by a number.

## Suggested next step (P0 continuation)

Re-run `p0-new-attacks`'s command at `--n-images 10+` (closer to real
`standard` scale) once there's time for a ~15-20 min run, specifically to
firm up the AVIF/WebP gate decision and get an actual flip misdecode rate
instead of "1 in 8". The pipeline/crop/rotate/perspective results don't need
a bigger n — they're structural, not statistical.

## Update (2026-09-21) — n=12 follow-up: `p0-new-attacks-n12/`

Re-ran the exact `p0-new-attacks` command with `--n-images 12
--rescale-images 12` (also bumped `--n-neg 8 --n-neg-rescale 4
--n-wrongkey-images 8`), same seed, 1 CPU, `--jobs 1`, 660 trials in 440s.
`pytest tests/` (212 tests, C2PA included) still green on this checkout
first — [đã xác minh].

- **`avif_60`**: 24/24 (both image kinds combined) detected, 0 misdecode,
  BER raw ≤0.004, comb 0.000, decode p50 ~30ms — same shape as `jpeg_75`/
  `webp_80` on this corpus. At n=4 this was "promising, not enough to gate
  on"; at n=24 it behaves identically to the already-gated rows. Reads as
  ready to move from `gate=False` to gated in `bench_attacks.py` — still on
  synthetic images only, so treat this as "gate it, but the first real-photo
  `standard` run is the actual confirmation," not a closed case.
- **`webp_80`**: 24/24, 0 misdecode — no change from pilot, already gated.
- **`flip_horizontal`**: still 0/24 detection (structural — mirrors the
  block raster, no oracle). Misdecode: 1/24 = 4.2% (was "1/8" at pilot
  scale), all on gradnoise, none on pink. Still thin for a rate anyone
  should quote, but the direction holds: flip produces occasional confident
  wrong answers, not just misses, and it's worth keeping the caveat in the
  README rather than promoting `crop_unaligned`/`rotate`'s stronger "0
  misdecodes ever" framing to flip.
- **`crop_unaligned_*`, `rotate_*deg`, `perspective_*`,
  `pipeline_resize_jpeg_crop`, `pipeline_heavy_repost`**: unchanged, 0/24
  detection each, 0 misdecode, `n/a` BER (no oracle) — confirms these are
  structural as the pilot already said; not worth more samples.
- **`ss` wrong-key acceptance**: 9/64 = 14.1% [upper bound 25.0%] this run,
  vs. the README's documented "~1-in-4". Point estimate is lower but the
  interval's upper bound lands right at 25% — consistent with the README
  claim at this n, not a contradiction, just not tight enough yet to say
  whether the true rate is nearer 14% or 25%.

Reports: `bench/results/p0-new-attacks-n12/report.md` (+ `records.jsonl`,
`summary.json/csv`, `meta.json` with the fingerprint/config).

**Not done in this pass**: no code changes (didn't flip `avif_60`'s
`gate=True` yet — that's a one-line change in `bench_attacks.py` but
touches the regression gate, left for the user to confirm first), no
real-photo (`--images DIR`) run, no README.md/bench/README.md text updates
to cite the n=12 numbers instead of the n=4 pilot ones.

## Update (2026-09-21, same day) — `avif_60` gated

Confirmed with the user. `bench_attacks.py` now registers `avif_60` without
`gate=False` (its data — 24/24 detected, 0 misdecode at n=12/kind — read the
same as the already-gated `jpeg_*`/`webp_80` rows, see above). `pytest
tests/` re-run after the change: still 212 passed. Also updated
`bench/README.md`'s Attacks section to match (removed the "measured but not
yet gated" carve-out for `avif_60`, folded the n=12 numbers into the
existing paragraph).

`p0-new-attacks-n12/records.jsonl` and `report.md` were generated *before*
this code change, so their stored `gate` field for `avif_60` originally
said `False` (matching the code at the time the trials ran). Since `gate`
is a classification label, not a measured value, and the underlying
detect/misdecode/BER numbers for `avif_60` are unchanged, that field was
patched to `True` in `records.jsonl` and `report.md` re-rendered with
`--report-only` to match the current code — no trials were re-run, no
numbers were altered, only which table row gets a † and which detection
rate counts as "gated" in the headline (72/72 → 96/96 gated at n=12,
including `avif_60`'s own 24/24). Flagging this explicitly since silently
editing a "measured" artifact is exactly the kind of thing this project's
own philosophy warns against — this edit only ever touched the `gate`
label, verifiable by diffing against the original run's config in
`meta.json` (unchanged) and the fact that `avif_60`'s detect/misdecode/BER
numbers above are identical to the pre-patch report.
