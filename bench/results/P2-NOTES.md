# ID-aware spread decoder (2026-09-21) — `wmcore.find_spread_spectrum_id`

Follows §2 item 2 of `HANDOFF-layered-repost-resistance-v18.md` ("decoder biết ID"): a fixed-length ID
(period 112 bits) with check characters lets the decoder skip period discovery, try the 112 phases, sum
the repeats, and accept the phase whose bytes are a valid ID. Everything below was run; what was not run
is under *Not measured*.

## 1. What changed

- `wmcore/spread.py`: `find_spread_spectrum_id(corrs)` — for each of the 112 phases, sum every repeat,
  hard-threshold the 80 payload bits, accept only a **canonical, check-valid** ID (`wmid.parse(t) == t`).
  The 32 header bits are deliberately unused. Cost: 0.7 ms at 4.8k blocks, 9 ms at 187k blocks (≈12 MP);
  the DCT that precedes it is ~1.5 s at 12 MP.
- `find_spread_spectrum_payload_or_id(corrs)`: ID decoder first, then the old text decoder. Wired into
  `extract_spread_spectrum` and `iter_spread_spectrum_payloads`, hence into `wmapp.extract` for both the
  `--ss-key` and the ledger-key paths. A payload that is not an ID falls through to the old decoder
  unchanged. The `--rescale` search is covered in §6 (added later the same day: a `decoder=` hook, wired into
  `extract`).
- Bench: decoder `ss_id`; payload `wmid` (a fresh check-valid ID per image, from the seed);
  `DECODER_PAYLOAD` runs `ss_id` only on `wmid`. `bench/experiments/id_decoder_probe.py` (new).
- Tests: `tests/test_spread_id_decoder.py` (14), one end-to-end test in `tests/test_id_ledger.py`.
  `pytest tests/`: **292 passed** (277 before + 15 new), 89 s on 1 CPU; after §6 (hook, `--resume`, resize test): **299 passed**, 138 s.

### A trap in the bench worth remembering
The bench's existing payload `id` (`K3M9QXR2T4`) is ID-*shaped* but **not check-valid**
(`wmid.parse` → `None`; the correct check characters for `K3M9QXR2` are `E9`). That was fine for P1's
payload-length runs, which only needed 10 bytes, but any decoder that verifies IDs scores 0% on it no
matter how good it is. Hence the new `wmid` payload; `id` is left as is so P1's records stay comparable.

### Why the old decoder loses here (mechanism, not just outcome)
`find_spread_spectrum_payload` bootstraps from the hard-thresholded raw correlations: it needs some
repeat whose 32 header bits are *all* right. At raw BER 20% that is 0.8^32 ≈ 0.08% per repeat; at 30%
≈ 1e-5. The sum over dozens of repeats is often clean long before that (P1-NOTES §1: "signal intact,
decoder loses sync"). `test_recovers_when_no_header_survives_but_the_sum_is_clean` builds this case
synthetically.

## 2. Measurements

Environment: 1 CPU, Python 3.12.3, numpy 2.4.4, OpenCV 4.13.0, Pillow 12.1.1, seed 20260919.
`meta.json` in each run dir records `wmcore_fingerprint 9b526cba70a9` — the code *before* the
`..._or_id` helper and the `extract` wiring were added. `find_spread_spectrum_id` itself did not change
between the runs and now; the runs exercise the decoders directly, not `extract`.

### 2.1 Synthetic images — `bench/results/p2-ss-id-synthetic/`
pink + gradnoise, 640×480 and 320×240, 12 images per cell (48 per attack), payload `wmid`, all 41 attacks
in the registry, decoders `ss` (text) and `ss_id` on the **same attacked image** (paired). 5552 trials,
190 s.

```
python3 bench/run_benchmark.py --suite standard --kinds pink,gradnoise --sizes 640x480,320x240 \
  --n-images 12 --decoders ss,ss_id --payloads wmid --n-neg 100 --n-wrongkey-images 24 --n-wrong-keys 8 \
  --out bench/results/p2-ss-id-synthetic --no-progress
```

Gated attacks: `ss` 765/768 (99.6%), `ss_id` **768/768** (100%). Over all 1968 paired trials: `ss` 889
correct, `ss_id` 906, **`ss_id` only 17, `ss` only 0** (exact one-sided sign test 2^-17 ≈ 8e-6; the images
share seeds across attacks, so read that as "consistent", not as an independent-trial p-value).

Rows where the decoders differ or a wrong string was returned (all other attacks: identical, 0 wrong):

| attack | gated | n | `ss` ok | `ss_id` ok | `ss_id` only | `ss` only | `ss` wrong | `ss_id` wrong |
|---|---|---|---|---|---|---|---|---|
| blur_1.0 | yes | 48 | 47 | 48 | 1 | 0 | 0 | 0 |
| blur_1.5 | no | 48 | 32 | 42 | 10 | 0 | 0 | 0 |
| crop_top_64 | yes | 48 | 46 | 48 | 2 | 0 | 2 | 0 |
| crop_topleft_64 | no | 48 | 0 | 0 | 0 | 0 | **18** | 0 |
| crop_unaligned_3 | no | 48 | 0 | 0 | 0 | 0 | 6 | 0 |
| flip_horizontal | no | 48 | 0 | 0 | 0 | 0 | 1 | 0 |
| jpeg_30 | no | 48 | 44 | 48 | 4 | 0 | 1 | 0 |

- **Wrong strings:** `ss` 28, `ss_id` 0. 17 of the 18 on `crop_topleft_64` are at 320×240. What `ss` returns
  there is long junk made of scrambled fragments of the true ID plus control characters (a left crop
  changes the row width, so the raster tiling no longer has period 112). The ID decoder rejects all of it.
  P0-NOTES reported 0 misdecodes on `crop_unaligned_*` for the `short` payload at 640×480; here 4/12
  (gradnoise 640×480) appear with an ID payload. Different payload and keys — the earlier statement is not
  contradicted, but "0 misdecodes" is payload-dependent.
- **Unmarked images (FPR):** `ss` 0/400, `ss_id` 0/400 (95% upper bound 0.9%). The analytic false-accept
  rate is ~1e-10 (per phase: header ignored, each of 10 bytes must land in the 32-character alphabet =
  2^-30, then 10 check bits = 2^-10; × 112 phases). 0/400 cannot verify a rate that small; it only fails
  to contradict it. `test_pure_noise_is_never_accepted` is a tripwire against loosening the rule.
- **Wrong key (exact payload recovered):** `ss` 64/384 = 16.7% [13.3–20.7%] (+1 wrong string), `ss_id`
  **112/384 = 29.2% [24.8–33.9%]**. Intervals do not overlap. By cell: 640×480 ~37% for `ss_id` (36/96) on
  both kinds; 320×240 gradnoise 36%, pink 5%.

### 2.2 Real photos — `bench/results/p2-ss-id-real/`
The same 7 OpenCV `samples/data` images as P0-bis (fruits, baboon, board, butterfly, aloeL, lena,
starry_night; downscaled to ≤1600 px), fetched from `raw.githubusercontent.com/opencv/opencv/4.x`. Still
classic CV test images, not repost content.

```
python3 bench/run_benchmark.py --suite standard --images <dir> --n-images 7 --decoders ss,ss_id \
  --payloads wmid --n-neg 7 --n-wrongkey-images 7 --n-wrong-keys 12 --out bench/results/p2-ss-id-real --no-progress
```

- Gated attacks: 112/112 for both. Ungated: blur_1.5 3→5 (`board.jpg`, `starry_night.jpg` gained), jpeg_30
  6→7 (`butterfly.jpg`); `ss_id` only 3, `ss` only 0. Wrong strings: 0 for both.
- Unmarked FPR: 0/28 for both (upper bound 12.3%).
- Wrong key: `ss` 35/168 = 20.8% [15.4–27.6%] (+2 wrong strings), `ss_id` 41/168 = 24.4% [18.5–31.4%]. **The
  intervals overlap heavily; n is too small to say whether the synthetic-image increase carries over.**

**The `board.jpg` miss from P0-bis (`ss`, blur_1.0, `short` payload) did not reproduce** with the `wmid`
payload — `ss` passes it here. That miss was payload/key-specific, so it is **not** evidence about `ss_id`.

### 2.3 Where the gain is, and how much room is left — `bench/results/p2-ss-id-probe.txt`
`bench/experiments/id_decoder_probe.py --sizes 640x480,320x240 --n 8 --images <dir> --pipeline 6`
compares `ss`, `ss_id` and `sum_exact` (an oracle: true phase, all 112 bits exact — says whether the
signal is intact; `ss_id` can exceed it because it ignores the header). Counts per attack are in the
file. In short: `ss_id` ≥ `ss` in every row and equals `sum_exact` in every row of this run; the largest gaps are JPEG q20 (640×480 5→15 of 16; 320×240 1→8; real 0→6 of 7) and blur 1.5–2.0.

**Room left:** of the 34 trials where `ss_id` still misses across the three corpora, **17 have ≤4 wrong
bits among the 80 payload bits at the true phase** (the rest have 6–16). (An earlier, unsaved scratch probe
with different seeds suggested most misses were ≥7 bits; the saved script does not reproduce that —
disregard the scratch reading.) Whether a soft or flip-few-bits decoder could turn those into
CRC-valid IDs — and at what false-accept cost, since every extra hypothesis spends some of the ~2^-40
budget — is **not measured**.

**Full pipeline** (`wmapp.watermark`, 640×480 gradnoise, grid text on, spread mark carrying the ID, PNG →
JPEG re-save; n=6): q50 6/6 vs 6/6; q30 6/6 vs 6/6; **q20 `ss` 0/6, `ss_id` 4/6**; q15 0/6 vs 0/6. `tests/test_id_ledger.py::
test_a_heavily_recompressed_copy_is_found_by_the_id_aware_decoder` pins one such case end to end
(negative control: with the ID decoder disabled that test fails).

## 3. What this does and does not show

- **Inside the documented range the gain is small** (3/768 on synthetic, 0/112 on photos). It is at the
  margin — JPEG ≤ q30, blur ≥ 1.5 — and in never returning a wrong string. It is not a new capability
  against rotation, perspective, unaligned crops or flips (all still 0%).
- Corpora are synthetic images and 7 classic test photos, n small on the photo side. Nothing here is a
  rate for real repost content (screenshots, social-media recompression, phone photos).
- The cost is in wrong-key readout, above. The ID returned is the true one, never another valid ID
  (probability ~2^-33 per attempt by the accept rule), so it does not misattribute; it does make
  `--ss-key` less of a secret than the README already said it was.
- Runtime is negligible against the DCT. Timings from the bench run are not quoted (the run shared the CPU
  with test runs).
- Not put in a regression baseline: no `bench/baselines/` exists in the repo yet.

## 4. Not measured

- ~~`ss_id` inside the `--rescale` search~~ — measured, §6.
- A soft / few-bit-flip ID decoder for the ≤4-wrong-bit misses (§2.3).
- Real repost content; larger real corpora; 1280×960 and up (the run sizes were 640×480 and 320×240).
- Left/right/unaligned crop recovery. Because an ID has a strong verifier, searching hypotheses (row
  stride × phase, and for pixel-unaligned crops 64 sub-block shifts) is safe against false accepts in
  principle; nothing has been built or measured. Real crops are rarely 8-aligned, so it would need the
  shift search too.

## 5. Reproduce

```
python3 bench/run_benchmark.py --suite standard --kinds pink,gradnoise --sizes 640x480,320x240 --n-images 12 \
  --decoders ss,ss_id --payloads wmid --n-neg 100 --n-wrongkey-images 24 --n-wrong-keys 8 --out /tmp/p2 --no-progress   # ~190 s
python3 bench/experiments/id_decoder_probe.py --sizes 640x480,320x240 --n 8 --images <dir> --pipeline 6                # ~1 min
```

## 6. Inside `--rescale` (measured later on 2026-09-21)

**Change.** `find_spread_spectrum_payload_rescaled(..., decoder=None)`: `decoder(corrs) -> text | None` reads the
payload at factor 1.0 and at every candidate width. The coarse sweep that proposes scales, the
two-widths-must-agree rule and the deadline are untouched. Its default stays the text decoder, so the bench's
`ss_rescale` remains the comparator. Wired into `extract` after the measurement below:
`extract_spread_spectrum_rescaled` defaults to `find_spread_spectrum_payload_or_id` (the `--ss-key` path; text
payloads unchanged), and the ledger-key loop passes `find_spread_spectrum_id` (it accepts only IDs, so a junk
text hit must not be able to end a key's search early). Bench decoders: `ss_id_rescale` (ID only),
`ss_or_id_rescale` (ID, else text). Also new: `run_benchmark.py --resume`, because background runs were being
killed at the end of a working session and the records file used to be truncated on start.

**Criteria, fixed before running:** paired detection on resized marked images at least equal to `ss_rescale`;
false-positive rate on never-stamped images (through identity / ×0.75 / ×1.25) not higher; no wrong strings;
runtime not materially worse.

### 6.1 Synthetic — `bench/results/p2-ss-id-rescale-synthetic/`
pink + gradnoise, 640×480 and 320×240, 6 images per cell (24 images), payload `wmid`, attacks
`identity`, `resize` (6), `resize_jpeg` (2) = 9, budget 20 s; three decoders on the same attacked image. Not
run: the two `pipeline` attacks (crop + resize), 1280×960. The run was killed once (771/816 records) and
finished with `--resume`, same arguments and seed; `wall_seconds` in `meta.json` covers only the resumed part.

| | `ss_rescale` | `ss_id_rescale` | `ss_or_id_rescale` |
|---|---|---|---|
| read, of 216 paired trials | 202 | **210** | **210** |
| wrong strings | 0 | 0 | 0 |
| unmarked images returning text (48 searches) | 0 | 0 | 0 (95% upper 7.4%) |
| median decode time, found resize trials | 1.73 s | 1.68 s | 1.69 s |

`ss_id_rescale` only: 8 pairs; `ss_rescale` only: 0. All 14 misses of `ss_rescale` were "not found", none a
wrong string. **The 8 pairs come from 3 images, all pink:** 320×240 seed …919 (5 attacks: ×0.9, ×1.25, ×1.5,
×0.5, ×1.4+JPEG), 640×480 seed …919 (×0.5, ×0.75+JPEG), 320×240 seed …920 (×0.5). Per attack: five gated
attacks +1 each (23→24 of 24), ×0.5 (ungated) 20→23. gradnoise images: no difference. So read this as
about three images out of 24 where the text decoder lost a mark that was there, not as eight independent
events. Timings are not benchmark-grade: light scripts ran alongside.

### 6.2 Real photos — `bench/results/p2-ss-id-rescale-real/`
The same 7 OpenCV images as §2.2 (up to 1282×1110), 9 attacks, 63 paired trials; wmcore fingerprint
`b62d785325a9`.
- Gated attacks: `ss_rescale` 53/56 (94.6%), `ss_id_rescale` 56/56, `ss_or_id_rescale` 56/56. Overall 56 / 59 / 59
  of 63. `ss_id_rescale` only: 3 pairs, on 2 images (`board.jpg` ×0.62 and ×0.75+JPEG; `fruits.jpg`
  ×0.75+JPEG); `ss_rescale` only: 0. Wrong strings: 0 for all.
- Still missed by all three: ×0.5 (ungated) on 4 of 7 photos — the search's known limit, unchanged.
- Unmarked: 0/21 for each (95% upper 16.1%); one search (`aloeL.jpg` ×1.25) ran out of time, identically for
  all three decoders — inconclusive, not a hit. Median decode time on found trials: 2.2 s for all three.

### 6.3 Synthetic and real unmarked images: the decoder is rarely asked — `bench/results/p2-ss-id-rescale-hypotheses.txt`
`bench/experiments/id_rescale_hypotheses.py` runs the real search on never-stamped images with a recording
decoder that asks both decoders about every correlation vector the search decodes (search path = the text
decoder's). 93 searches (36 at 640×480, 36 at 320×240, 21 on the real photos) made **147 decode attempts**:
36 for the 36 searches at 640×480 (the sweep proposed no candidate at all; only the plain, unresized attempt
ran), 51 at 320×240, 60 on the photos. The text decoder returned something on 0 of them and the ID decoder an ID
on 0; two real-photo searches hit the time budget. On these images the coarse sweep is what limits false
positives — the decoder is consulted about 1.6 times per unmarked search — so 0/48 and 0/21 (§6.1–6.2) say more
about the sweep than about the decoders. **That does not carry over to periodic images: see §6.6.** (An earlier
draft of this section concluded that the decoder barely matters for false positives; §6.6 is why that is retracted.)

### 6.4 Through the full pipeline
Stamped with `wmapp.watermark` (grid on, spread mark carrying the ID), 320×240 pink, seeds 1–24, resized ×1.25 and
×0.75 (bicubic, PNG), `find_spread_spectrum_payload_rescaled` with the text decoder vs the ID decoder: the
text decoder found nothing and the ID decoder read the ID for seeds 3, 8, 19 at ×1.25 and 8, 19 at ×0.75 (3 of 24
images; the reverse direction — text decoder reads, ID decoder does not — was not checked in this search). `tests/test_id_ledger.py::
test_a_resized_copy_is_found_by_the_id_aware_resize_search` (seed 8, both scales, via the ledger and via
`--ss-key`) pins it; with the wiring forced back to the text decoder it fails (negative control).

**A flake found and fixed while writing this.** `wmapp.watermark` draws both the ID and the `ss_key` at random,
and the ID decoder is not universal: for pinned `ss_key`s 20260921–30 at JPEG q20 (ID drawn per key) it read 9
of 10 (the text decoder 0 of 10); for the resize case above, with key 20260921 neither decoder read either
scale, with 20260922 only the ID one did. My first versions of the q20 test (§2.3) and of this one did not pin
key and ID, so they would have failed on some runs — the resize test did, once, in a full-suite run. Both now
pass `ss_key=` and `id_value=` (q20: 20260921 / `WR5TT382BJ`; resize: 20260922 / `V2ACCRPDPV`). The
probe numbers in §2.3 (q20 4/6 vs 0/6) and this section's "3 of 24" already reflect random keys.

### 6.5 Not measured
Crop + resize (`pipeline` attacks); 1280×960 and up (where the grid-corner anchoring of P0-NOTES §1 already
limits the search); ×0.5; real repost content; the reverse direction of §6.4 (text decoder reads, ID decoder
does not). Wrong-key readout inside the search only at 320×240 (n=100) and 640×480 (n=16), §6.7. False-positive
evidence is 0/48 and 0/21 searches, 147 decode attempts, and the 10 hand-made periodic patterns of §6.6 — not a
rate for real images.

### 6.6 Periodic (never-stamped) images — `bench/results/p2-ss-id-rescale-structured.txt`
`id_rescale_hypotheses.py --sizes "" --structured 640x480 --id-search`: 10 periodic patterns, each with a little
noise (checkerboards at 8, 16 and an off-grid 12 px, horizontal/vertical stripes at 8 px, 3 px stripes, 16 px grid
lines, bricks, dots, diagonal stripes), 3 attacks each (identity, ×0.75, ×1.25) = 30 searches, budget 20 s.
- The sweep proposes many candidates here: **660 decode attempts** (22 per search, against ~1.6 on the natural
  images). The text decoder returned something on **92** of them — junk. The ID decoder returned an ID on **0**.
- The text-decoder search **accepted junk in 2/30 searches** (6.7%, 95% CI 0.8–22.1%): `gridlines16` at identity and at
  ×1.25. Both came from the *plain, unresized* decode (`attempts=1`, factor 1.0), not from the resize search: strings
  of NUL and other control bytes (valid UTF-8, so the text decoder accepts them; the plain path has no printable
  rule — the search's own two-agreeing-printable-widths rule is what kept the candidate-stage junk out).
- The **ID-only search, run for real on the same 30 images, accepted 0** (95% upper bound 11.6%).
- Three ×1.25 searches (`stripes3`, `bricks32x16`, `diag_stripes`) hit the time budget: inconclusive, and not
  recorded per decoder.
Consequences. The false positives are the *plain* text decoder's, so they exist in a plain `extract --ss-key`
too; the wiring did not create them. The ledger path is unaffected (it accepts only a check-valid ID: the
`wmid.parse` filter and the ID-only search). The `--ss-key` path reads "an ID, else text", so on a periodic
image with no mark it can still report junk text, as it always could. A printable-characters filter on the plain
path would remove these two, but text payloads may legitimately hold control characters, so it is a proposal, not
done and not measured. Caveats: ten hand-made patterns, one size, n = 30 — evidence that the text decoder's false
positives exist and the ID reader has none *here*, not a rate.

### 6.7 Wrong keys on resized copies — `bench/results/p2-ss-id-rescale-wrongkey.txt`
`bench/experiments/id_rescale_wrongkey.py`: marked images resized, decoded with random wrong keys through the text
search (`ss_rescale`) and the ID search (`ss_id_rescale`), run separately. "Read" = the true payload came back.
| | text search | ID search |
|---|---|---|
| 320×240, true-key control (20) | 15/20 | 20/20 |
| 320×240, wrong keys (100 = 10 images × ×0.75, ×1.25 × 5 keys) | 19 read (19.0%, CI 12.5–27.8%) | 23 read (23.0%, 15.8–32.2%) |
| 640×480, wrong keys (16 = 4 images × ×0.75 × 4 keys) | 2 read (12.5%, CI 3.5–36.0%) | 3 read (18.8%, 6.6–43.0%) |
| pooled (116) | 21 (18.1%, 12.2–26.1%) | 26 (22.4%, 15.8–30.8%) |
No other text came back in any wrong-key trial (0 for both), and no search hit the budget. A wrong key still reads the
true payload about one time in five through the resize search, for both decoders; the difference between them
(4–5 points) is inside the intervals. The trials are not independent (5 wrong keys share one image), and 640×480 has
only 16 per decoder. It does not show that the search leaks *more* than plain decoding (§2.1: 16.7% text / 29.2% ID).

### 7. Printable filter added to the plain decoder (measured 2026-09-22) — `bench/results/p2-printable-filter.txt`
§6.6 traced 2/30 junk accepts on periodic images to the *plain, unresized* decode (`find_spread_spectrum_payload`
itself), which had no printable rule — only "valid UTF-8". `bench/experiments/plain_decode_printable.py` measures
that path directly (no `--rescale` at all): 10 periodic patterns × 3 sizes (640×480, 320×240, 1280×960) × 3 states
(identity, resized ×0.75, resized ×1.25) = 90 plain decodes on never-stamped images, plus 4 payloads containing
`\t \n \r` × 3 conditions (raw, JPEG q95, JPEG q75) = 12 round-trip trials on real embedded payloads.
- **Before:** 8/90 plain decodes on periodic images returned junk (NUL runs, repeated 0x55 bytes, other control
  bytes — valid UTF-8, so the old check accepted them). 12/12 control-character payloads round-tripped (no filter
  yet, so this is the pre-change baseline, not evidence of anything).
- **Change:** `_printable_payload` (was rescale.py-only: printable, or one of `\t \n \r`) moved to spread.py and
  applied inside `find_spread_spectrum_payload` itself, so every caller gets it — the plain decode, the rescale
  search's factor-1.0 fast path (which calls the same function), and `iter_spread_spectrum_payloads`. rescale.py
  now aliases the one definition instead of keeping its own.
- **After:** 8/90 → **1/90** junk on periodic images. The one survivor (`dots8` / resize ×0.75, `'?'`) is a single
  printable ASCII character — indistinguishable from a genuine 1-character payload by a printable-only rule; not
  closed, a known residual. 12/12 control-character round-trips unchanged.
- New tests: `tests/test_spread_spectrum.py::test_roundtrip_tab_newline_cr` (pins the round-trip) and
  `::test_other_control_characters_do_not_roundtrip` (pins the actual behavior change: a payload holding a control
  character outside `\t \n \r`, e.g. a bell or embedded NUL, no longer round-trips — narrow but real; not measured
  as a rate, just confirmed to happen). `pytest tests/`: 301 passed (299 + 2 new), run in four chunks.
Not done: the surviving 1-char false-positive path (would need a length/entropy heuristic, not just printable);
a rate for real images (still 10 hand-made patterns, one seed); whether a length-independent statistic would tell
a real short payload from a lucky short junk match.

**Addendum (2026-09-22, during P3): the residual is seed-dependent.** `bench/experiments/flip_text_probe.py` part B
ran the same 10 periodic patterns × 3 sizes × 3 states on a different seed set (base seed +3 000 000 instead of
+2 000 000) through the same, already-filtered plain text decoder: **2/90** unmirrored images returned junk
(`'%'` at 1280×960 gridlines16 / ×0.75, `'"*'` at 1280×960 bricks32x16 / identity — one and two printable characters),
not 1/90. So "8/90 → 1/90" above is the figure for that seed set; over the two seed sets tried the residual is 1–2 of
90 (n is far too small for a rate), and it is short printable junk: the filter removes the control-byte runs, not short strings. The 2 don't come
from mirroring (the unmirrored orientation is the old reader's). Still not measured: a length/entropy rule, and any
rate on real images (`bench/results/p3-flip-text-probe.txt`).
