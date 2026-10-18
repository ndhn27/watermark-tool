# P1 measurements (2026-09-21) — read before quoting anything about `--id`

What was measured while building `--id` (short watermark ID + ledger), and what it does and
doesn't support. All of it is on **synthetic** images (`pink`, `gradnoise` from
`bench/bench_corpus.py`), default `--ss-alpha 12`, `--jobs 1`, one CPU. None of it has been run on
real photos: treat every number as "the direction on this corpus", not a figure to publish.
The `records.jsonl` / `report.md` in each directory are the raw evidence.

## 1. Does a short ID make the `spread` mark more robust? — only against long payloads on small images

Question: the handoff's rationale (b) said a short payload → more repeats → more robust. Measured with
the bench's embed-only path, decoder `ss` (plain, no resize search), payload `id` = a 10-byte watermark
ID (112 bits on the wire), `short` = `@yourhandle` (11 B, 120 bits), `medium` = 45 B, `long` = 99 B
(824 bits). Counts are correct-payload detections out of n=16 (8 images × 2 kinds); "w" = a *wrong*
text returned.

**640×480** (`p1-payload-length/`): every gated attack is 16/16 for all four payloads
(identity, jpeg_60, jpeg_50, noise_5, noise_10; blur_1.0 is 15–16/16). Outside the documented range,
`jpeg_30`: id 11/16, short 16/16, medium 11/16, long 9/16 — **not monotone in length** (the 10-byte ID
is no better than the 45-byte payload, and it does worse than the 11-byte handle of nearly the same size:
11/16 vs 16/16, Fisher p ≈ 0.04 among many comparisons — could be chance, could be an effect of the
header bit pattern on the header-window scan; not investigated), so this is not a payload-length effect. Those misses are the
bench's "signal intact – decoder loses sync" rows: oracle-combined BER is 0.000 while the decoder
fails, i.e. the failure is in the header-window scan, not in the signal.

**320×240 and 480×360** (`p1-payload-length-small/`, 1200 and 2700 blocks):

| 320×240 | id | short | medium | long |
|---|---|---|---|---|
| identity | 16/16 | 16/16 | 16/16 | 12/16 (w1) |
| jpeg_60 | 16/16 | 16/16 | 16/16 | 10/16 (w2) |
| jpeg_50 | 16/16 | 16/16 | 16/16 | 11/16 (w1) |
| blur_1.0 | 16/16 | 16/16 | 14/16 (w1) | 9/16 (w3) |
| crop_top_64 | 16/16 | 16/16 | 15/16 | **0/16** |

At 480×360 only `medium` and `long` slip, slightly (blur_1.0: medium 15/16, long 14/16; jpeg_60: long 15/16); the rest is 16/16.

Reading: the ID is **not** more robust than the short handle the tool used by default — same size, same
results everywhere. It is much more robust than a *sentence-length* `--invisible-text` (or
`--copyright`-style payload) on *small* images, where the long payload has ~1.5 repeats and fails even
unattacked (and returns wrong text a few percent of the time). The wrong-text ("w") counts are worth noticing: a free-text payload has
nothing to reject a wrong decode with, while for an ID payload the check characters do (one `id` misdecode
at jpeg_30 on 640×480 would almost certainly have been refused by `extract`).

## 2. Does `--grid-id` hurt the hidden mark? — no effect detected

`bench/experiments/grid_id_interference.py` (real pipeline, `wmapp.watermark`, then the same decoder
`extract` uses, key known). Configs: A legacy payload + short grid text; B `--id` + short grid text;
C `--id --grid-id`; D legacy payload + a grid text as long as C's.

- **640×480, pink+gradnoise, n=16 per cell** (`p1-grid-id-640x480/`): A, B, C, D are all **16/16** after
  identity, JPEG q60, JPEG q40 and a 0.75× resize decoded with `--rescale`.
- **1280×960 gradnoise, resize 0.75× + rescale only, n=10** (`p1-grid-id-1280x960-gradnoise/`):
  A 3/10, B 3/10, C 3/10, D 0/10 (Wilson 95%: [11%, 60%] for the 3/10 cells, [0%, 28%] for D).
  A/B/C are indistinguishable; D vs A is not significant at this n (Fisher p ≈ 0.2), and C, which has the
  same long grid text as D, matches A — so there is no consistent grid-text-length effect.

An earlier n=3 look at 1280×960 (B 2/3, C 1/3, D 0/2) suggested C and D were worse; n=10 did not
confirm it. That is the reason the n=10 run exists; don't cite the n=3 one.

**A finding that is not about `--id`:** the baseline itself (A) recovers only 30% of 0.75×-resized
`gradnoise` images at 1280×960, whereas the same resize on 640×480 images is 16/16 (and `pink` at
1280×960 was 9/9 in a small check). The resize search's success depends on the image kind and size more
than the P0 pilot (640×480 only) showed. Not investigated; worth a P0 follow-up before README numbers
say "resize 0.75× is handled" without a size.

## 3. Wrong keys still decode ~1 time in 4 — why `extract` validates the ID

In a 501-key check on one 1920×1080 synthetic image (right key + 500 random ones), roughly a quarter of
the wrong keys decoded the *correct* payload and a few returned near-miss junk (`K3M)QXR\x12T4`,
`K3M9QXR2T4\x00\x00\x00\n`, `0`). This matches the README's "~1 in 4" (and the p0-n12 pooled 9/64).
Counted by eye, not scripted. It is why the ledger-key scan accepts only a payload that parses as a
well-formed ID (check characters), and why the ID reported may come from a key other than the recorded
one — harmless, because the payload is the same.

## 4. Speed of scanning the ledger's keys

`wmcore.spread_spectrum_block_correlations_multi` shares one DCT across keys: 501 keys on a
1920×1080 image took 1.0 s total (vs ~0.22 s *per key* with the single-key function), and its output is
bit-identical to the single-key function (`max abs diff 0.0`). The 12 MP single-key cost is ~1.5 s, so
the shared version is what makes "try every key in the ledger" feasible. `--rescale` with several keys
can't share that work: it runs key by key against ONE shared time budget and reports "inconclusive"
(with how many keys it reached) rather than run for hours.

## 5. Proposal, not built: an ID-aware decoder

Because the ID has a fixed length, a decoder could skip the header scan: known period (112 bits), try
each of the 112 phases, combine repeats, accept the phase whose bits form a valid ID (CRC). Section 1's
`jpeg_30` rows (BER-combined 0.000, yet 5 of 16 not decoded) are the ceiling that would move.
Cheap to try, easy to measure with the existing oracle BER; not started.

## Reproduce

```
python3 bench/run_benchmark.py --suite standard --kinds pink,gradnoise --sizes 640x480 --n-images 8 \
  --payloads id,short,medium,long --decoders ss \
  --attacks identity,jpeg_60,jpeg_50,jpeg_30,resize_0.75,noise_5,noise_10,blur_1.0 \
  --n-neg 2 --n-wrongkey-images 2 --n-wrong-keys 2 --out bench/results/p1-payload-length --no-progress
python3 bench/run_benchmark.py --suite standard --kinds pink,gradnoise --sizes 320x240,480x360 --n-images 8 \
  --payloads id,short,medium,long --decoders ss \
  --attacks identity,jpeg_75,jpeg_60,jpeg_50,noise_5,blur_1.0,crop_top_64 \
  --n-neg 2 --n-wrongkey-images 2 --n-wrong-keys 2 --out bench/results/p1-payload-length-small --no-progress
python3 bench/experiments/grid_id_interference.py --n 8 --out bench/results/p1-grid-id-640x480
python3 bench/experiments/grid_id_interference.py --n 10 --kinds gradnoise --size 1280x960 \
  --attacks "resize_0.75 (+rescale)" --out bench/results/p1-grid-id-1280x960-gradnoise
```
(The 640×480 payload table above pools the two image kinds; the report.md files list them separately.
The timing columns in these reports were produced on a shared 1-CPU sandbox and aren't meant to be quoted.)
