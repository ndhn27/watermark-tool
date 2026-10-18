# Mirrored copies (2026-09-22) — `wmcore/orient.py`

Step 2/5 of the repost-resistance plan: "at spread decode time, also try flipping left-right / top-bottom;
measure with the bench before and after". Everything below was run; what was not run is under *Not measured*.
Environment: 1 CPU, Python 3.12.3, numpy 2.4.4, OpenCV 4.13.0, Pillow 12.1.1, seed 20260919.

## 1. Before (no mirror-aware code)

Attacks added to the bench first (`bench_attacks.py`, family `flip`, each with an oracle geometry so BER says
whether the signal is still there): `flip_vertical`, `flip_both` (= 180° turn), `flip_horizontal_jpeg_75`,
`flip_vertical_jpeg_50`; the older `flip_horizontal` (family `geometric`, no oracle) is unchanged. Also
`flip_horizontal_resize_0.75` and `flip_vertical_resize_1.25` (family `pipeline`).

Synthetic — `p3-flip-before-synthetic/`: pink + gradnoise, 640×480 and 320×240, 12 images per cell = 48 images,
payloads `wmid` (a check-valid ID), `short`, `medium`, decoders `ss` (text) and `ss_id`, 5 flip attacks + identity +
JPEG q75, 100 unmarked images, wrong-key trials. Real photos — `p3-flip-before-real/`: the 7 OpenCV `samples/data`
images of P0-bis/P2 (fruits, baboon, board, butterfly, aloeL, lena, starry_night; ≤1282×1110).

| | flips (5 attacks) read | wrong strings on flips |
|---|---|---|
| `ss`, ID payload, synthetic | **0/240** | 3 (1 left-right, 2 both-ways) |
| `ss_id`, ID payload, synthetic | **0/240** | 0 |
| `ss`, `short` / `medium`, synthetic | 0/240 each | 5 / 0 |
| `ss`, `ss_id`, ID payload, 7 photos | 0/35 each | 0 |
| `ss`, `short` / `medium`, 7 photos | 0/35 each | 0 / 0 |

The oracle BER on the mirrored copies that have an oracle (`flip_vertical`, `flip_both`, the two JPEG ones) is
**0.000** in `report.md` ("signal intact - decoder loses sync"): the mark is not damaged by a mirror, the decoder just
never looks at the image the right way round.

Mirror + resize — `p3-flip-before-rescale/`: `ss_id_rescale` (the resize search, ID reader), 16 marked images
(pink + gradnoise, 640×480 and 320×240), budget 20 s: `flip_horizontal_resize_0.75` **0/16**,
`flip_vertical_resize_1.25` **0/16**; plain resizes 16/16 and 16/16 (control). A search that finds nothing takes a
median of 5.8 s (mirror + ×0.75) and 10.8 s (mirror + ×1.25), max 16.7 s, on these sizes; on never-stamped images the
median is 5.5–10.5 s. (The run was killed once and finished with `--resume`;
`wall_seconds` in `meta.json` covers only the resumed part.)

## 2. What changed

- `wmcore/orient.py` (new): `find_spread_spectrum_id_oriented(img, key)` and
  `find_spread_spectrum_payload_or_id_oriented(img, key)` → `OrientedMatch(text, orientation)`, orientation ∈
  `none | h | v | hv`. Undo each of the four mirrorings on the luma plane (exact, any image size), compute the block
  correlations, decode.
- **What counts as a hit.** Mirrored orientations are searched for a **watermark ID only** (check-valid, ~1e-10
  false-accept per orientation). The text decoder runs only on the image as given. The ID search goes over **all**
  orientations before the text decoder is consulted: a retry loop that consulted the loose text decoder per
  orientation could return junk from the unmirrored orientation before the mirrored one, where the real ID is, was
  tried (this design bug was caught in the previous session before shipping; `tests/test_orient.py::
  test_the_id_is_searched_in_every_orientation_before_any_text_is_accepted` forces junk text everywhere and pins the
  order; with the per-orientation-text ordering the same setup returns the junk, checked).
- `wmcore/invisible.py`: `extract_spread_spectrum_oriented`, `iter_spread_spectrum_payloads_oriented` (one pass per
  orientation over all ledger keys, sharing one DCT per pass; the unmirrored pass first and exactly as before).
  `extract_spread_spectrum` is untouched.
- `wmapp/extraction.py`: `--ss-key` path and ledger path use the mirror-aware readers; `ExtractResult.flipped`
  (`"h" | "v" | "hv" | None`), a report line ("recovered after flipping the image back: this copy had been mirrored
  left-right"), and `flipped` in `to_dict()`. With `--rescale`, the three mirrorings are checked first (a few block
  DCTs), then the resize search runs exactly as before on the image as given. `keys_tried` still counts keys, not
  (key, mirroring) attempts.
- Bench: decoders `ss_flip` (ID in any orientation, else text on the image as given) and `ss_id_flip`; family
  `flip` out of scope for `ss`, `ss_id`, `lsb`; in scope for `ss_flip` only on the `wmid` payload
  (`IN_SCOPE_ONLY_FOR_PAYLOAD`), so the headline is not dragged down by a case that is a documented limit.
- Tests: `tests/test_orient.py`, 23 tests (see §5).

## 3. After — same images, same seed

`p3-flip-after-synthetic/`, `p3-flip-after-real/`: decoders `ss, ss_id, ss_flip, ss_id_flip` on the same attacked
images (paired). **The `ss` and `ss_id` columns reproduce the "before" runs exactly** (record-for-record outcome
counts equal, both corpora): nothing about the old readers changed.

| ID payload, 5 flip attacks | `ss` | `ss_id` | `ss_flip` | `ss_id_flip` |
|---|---|---|---|---|
| synthetic (240 trials) — read | 0 | 0 | **240** | **240** |
| synthetic — wrong strings | 3 | 0 | **0** | 0 |
| 7 photos (35 trials) — read | 0 | 0 | **35** | **35** |

- Per attack, synthetic: 48/48 for each of `flip_horizontal`, `flip_vertical`, `flip_both`,
  `flip_horizontal_jpeg_75`, `flip_vertical_jpeg_50` — including the JPEG ones.
- Unmirrored images: identity + JPEG q75, ID payload, 96/96 (synthetic) and 14/14 (photos) for all four decoders.
  Text payloads (`short`, `medium`) unmirrored: `ss_flip` = `ss`, trial for trial (96/96 each, synthetic).
- **Unmarked images (FPR):** `ss_flip` 0/400 and `ss_id_flip` 0/400 synthetic (95% upper bound 0.9%); 0/28 and
  0/28 photos (12.3%). Same as `ss`, `ss_id`.
- **Wrong key:** `ss_id_flip` 112/384 read the true ID = `ss_id` 112/384 (29.2%); `ss_flip` 113/384 accepted, 112
  exact. Photos: 41/168 for both ID readers. The mirrored orientations did not add any wrong-key readouts; the
  figure is the ID decoder's own leak, already documented in P2-NOTES §2.1 (`ss` alone is 64/384 = 16.7%).
- **Free-text payloads on mirrored copies: still 0/240 (`short`, `medium`) and 0/35 — by design (§4).** The old
  wrong strings on `short` (5/240) are still there, unchanged.
- **Runtime** (bench p50, 1 CPU, run alone): marked unmirrored ID `ss_id` 24 ms → `ss_id_flip` 25 ms (the first
  orientation answers); unmarked `ss` 18 ms → `ss_flip` 66 ms (synthetic 640×480/320×240 mix); photos 34 → 115 ms.
  From `flip_text_probe.py` part D: 640×480 unmarked 38 → 131 ms, 1280×960 195 → 541 ms, 4000×3000 1.8 s → 5.2 s
  (≈3–4×). The "marked, unmirrored" rows in part D used a payload that is not a valid ID (the old bench payload
  `K3M9QXR2T4`), so they time the text path, not the ID path — the 25 ms above is the ID case.

### 3.1 Re-run of §3 on the shipped code (2026-09-22, later the same day)
Both runs of §3 repeated with the exact commands of §7, on the code as packaged (`wmcore_fingerprint 77001f8bb294`;
the runs kept in `p3-flip-after-*` have `43bd87e6dee4`, taken before docstring-only edits). Result: **0 differences,
record for record** (same image, decoder, attack, payload, outcome), on both corpora — 5968 synthetic and 1197 photo
records. So 240/240 and 35/35 (ID payload, mirrored), 0/400 and 0/28 (unmarked), and `ss_id_flip` 112/384 = `ss_id`
112/384 (wrong key) reproduce on the shipped code. Timings in `meta.json` of that re-run are not quoted (single CPU,
shared with a `pytest` run just before). `pytest tests/` on the same tree: 324 passed in 143 s, one command.

## 4. Decision: mirrored TEXT payloads are not searched — `p3-flip-text-probe.txt`

`bench/experiments/flip_text_probe.py` measures what running the text decoder on mirrored orientations would do,
same images:
- **Gain:** on mirrored `short`/`medium` payloads (288 trials), the text decoder after undoing the *right* mirroring
  reads 283/288 (2 none, 3 wrong strings). In the 3 wrong orientations of those images it returned something on
  1/864.
- **Cost:** never-stamped natural-ish images (100; 400 orientation-decodes): 0 returned. Never-stamped **periodic**
  images (the 10 patterns of P2-NOTES §6.6 × 3 sizes × 3 states = 90 images): 5 junk strings across the four
  orientations — 2 unmirrored, 3 on mirrored orientations (short, 1–4 characters: `'DPDT'`, `' '`, `'I$'`).
  Searching all four orientations would have raised the number of periodic images that return junk from 2/90 to 5/90.
- **Shipped decoder vs the old one on those 90 periodic images:** old 2 returned, mirror-aware 2 returned, 0
  different.

The user's rule is "keep a change only if it adds readability without adding false positives or wrong strings", so
the text decoder is not run on mirrored orientations. The ID path adds ~1e-10 per orientation and measured 0.
Consequence: a mirrored copy of a *free-text* mark is not recovered. The tool's direction is `--id` + ledger, so this
covers the main path; if text payloads matter, the fix would be a stricter rule for text on mirrored orientations
(minimum repeats and an agreement statistic), not measured.

## 5. Tests

`tests/test_orient.py` (23): ID read in all four orientations at 643×477 (not a multiple of 8); mirrored + JPEG q75;
the old reader cannot read a mirrored ID; text payload read unmirrored and *not* read mirrored; unmarked image
matches nothing; the ordering rule (forced junk text); `orientations=("none",)` equals the old reader for an ID, a
text payload and an unmarked image; `unflip` is its own inverse and rejects unknown orientations; `extract` with the
ledger key (h, v, hv: `flipped`, the report line, `to_dict`, `keys_tried == keys_available`), with a given key, with
`--rescale` (mirroring found first, `rescaled_by` None), unmirrored copies report no mirroring, the right key among 3
runs on a mirrored copy, mirrored text via `extract` not found. Negative controls run by hand: per-orientation-text
ordering returns the junk; with mirroring forced off in `extract`, the mirrored copy is not found.

`pytest tests/` (two chunks, ~50 s + ~100 s): **324 passed** (301 + 23), incl. `test_frontends_parity.py`. Not
skipped. The two `bench` runs' `wmcore_fingerprint`s (`b4df979c2348` before, `43bd87e6dee4` after) are the code at
run time; afterwards only docstrings in `orient.py` (and the READMEs) were edited.

## 6. Not measured / not done

- **Mirror + resize.** `--rescale` does not search mirrored copies (0/16 before, unchanged). Adding it is not free: a
  search that finds nothing takes 5–10 s per image here, so four orientations would be 20–40 s against the default
  20 s budget, turning today's clean "not found" into "inconclusive" for unmarked images. Cheaper routes exist
  (sweep tier 1 over the four orientations first, run the full search only where a peak shows) but are not built.
- 90° rotations and transposes (the other four elements of the same symmetry group); the `lsb` layer (a mirror
  reverses the bit stream; LSB does not survive recompression anyway); video (step 3).
- Real repost content (photos that were actually mirrored by an app and re-uploaded); larger real corpora — 7 classic
  test photos, ≤1282×1110; images above 4000×3000 for timing.
- Mirror + other attacks beyond JPEG (blur, noise, crops): a crop after a mirror moves the grid and is as unsupported
  as an unmirrored crop.
- A rate for text payloads: 288 trials, 5 marked-image failures; the periodic-image numbers are 10 hand-made
  patterns, one seed set.

## 7. Reproduce

```
python3 bench/run_benchmark.py --suite standard --kinds pink,gradnoise --sizes 640x480,320x240 --n-images 12 \
  --decoders ss,ss_id,ss_flip,ss_id_flip --payloads wmid,short,medium \
  --attacks identity,jpeg_75,flip_horizontal,flip_vertical,flip_both,flip_horizontal_jpeg_75,flip_vertical_jpeg_50 \
  --n-neg 100 --n-wrongkey-images 24 --n-wrong-keys 8 --out /tmp/p3 --no-progress --resume   # ~5 min: run twice
python3 bench/run_benchmark.py --suite standard --images <dir of the 7 photos> --n-images 7 --decoders ss,ss_id,ss_flip,ss_id_flip \
  --payloads wmid,short,medium --attacks <same> --n-neg 7 --n-wrongkey-images 7 --n-wrong-keys 12 --out /tmp/p3r   # ~2 min
python3 bench/experiments/flip_text_probe.py                                                                        # ~2 min
```
The "before" runs were made with the same commands and `--decoders ss,ss_id` before `orient.py` existed.
