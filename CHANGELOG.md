# Changelog

## v6 — 2026-10-18

The biggest change since v5: the tool grew from "a visible grid plus an LSB backup" into a **stack of independent
layers** (visible grid, spread-spectrum mark, metadata, C2PA) that can be read back **from the file alone**, even
after the copy was recompressed, cropped, resized or mirrored. v5 → v6 takes `cli/` from about 1,400 to about 6,900 lines of
Python, grows the web tool from ~1,000 to ~1,400 lines, adds a benchmark harness, and takes the test suite from 45 tests
to 413 (all passing on Python 3.12 with `c2pa-python` 0.37.10; the JS tests in `tests/js` are separate).

### Added

**Invisible mark that survives JPEG**
- `--invisible --invisible-mode spread` (`--ss-key`, `--ss-alpha`): a spread-spectrum mark in the block-DCT domain.
  Unlike the LSB mark it survives JPEG recompression. Default strength `12.0` for images.
- It works for **video**: embedded in every frame inside the same worker pool as the grid, strength `20.0`
  (measured against real ffmpeg/x264 at CRF 23), key rotating every `--jitter-refresh-seconds`.

**Watermark ID and ledger**
- `--id` (`--id-value`, `--grid-id`, `--ledger`): one short ID per output, e.g. `K3M9-QXR2-T4` (40 random bits +
  2 check characters), written into every layer that can carry it.
- A local SQLite ledger (`~/.watermark-tool/ledger.sqlite3`, created `0600`) stores the SHA-256 of the input and the
  finished output, the key, the strength and the settings. The row is written *before* stamping, so a crash can't
  release an image the ledger doesn't know.
- `cli/wm_ledger.py` with `show`, `list`, `find`, `keys`.

**`extract` rebuilt**
- Needs only the file: with no `--ss-key` it tries every key in the ledger (newest first), accepts only a
  well-formed ID, and reports the ledger record and whether the file is byte-identical to the recorded output.
- Reads each layer (LSB, spread-spectrum, metadata, C2PA) **on its own**; `--layers` prints one status per layer
  (`found` / `text` / `none` / `inconclusive` / `skipped`). A warning is printed when two layers read different IDs.
- **Crop resync** (on by default; `--no-resync`, `--resync-budget-seconds`): re-finds the original 8×8 block grid and
  row length from the copy alone, so cropped screenshots and mirrored copies decode.
- **`--rescale`** (`--rescale-budget-seconds`): also reads copies that were cropped *and* resized; plus a rescale
  detector for copies resized after stamping. Ledger keys are swept in rounds, newest first, so a copy of a recent stamp
  reads in a few seconds however long the ledger is.
- Measured (details in `bench/results/CROP-NOTES.md`): crop + resize read on 84 of 84 trials on 7 real photos, crops
  down to ~25% of the area on real photos, 0 false IDs on 55 mark-free images.

**C2PA / Content Credentials**
- `--c2pa` (`--c2pa-cert`, `--c2pa-key`, `--c2pa-alg`, `--c2pa-title`) signs images **and** video;
  `cli/c2pa_verify.py` reads the manifest back. The watermark ID travels in the manifest's CreativeWork assertion.

**Video**
- Content-adaptive placement now tracks across frames (`--adaptive-track`, `--adaptive-redetect-seconds`) with
  scene-cut detection (`--scene-cut-detect`, `--scene-cut-threshold`).

**Web tool**
- `cli/serve.py`: optional local server (loopback only, standard library only). The page detects it and offers a
  **Full pipeline (server)** switch that renders with the same code as the CLI. Controls are built from `/api/schema`.

**Benchmarks**
- `bench/`: attack harness, corpus, metrics, scheme comparison, reports, and the experiments behind the README's
  numbers. `bench/repost_roundtrip.py` (`prepare` / `check`) stamps your own files, then reads back the copies you
  download from each platform.

### Changed
- `cli/wmcore.py` is now the `cli/wmcore/` package. `import wmcore` and the flat `wmcore.*` names are unchanged.
- New shared application layer `cli/wmapp/` (options, pipeline, extraction, ledger, limits, events): the CLI, the
  Python API and the server run the same code (`tests/test_frontends_parity.py` checks the pixels match).
- `cli/watermark.py` is now a thin shell over `wmapp`.
- `README.md`: threat model, per-layer survival notes, C2PA, ledger, crop resync and repost round-trip sections.
- `requirements.txt`: adds `c2pa-python` (pinned, see below) and `cryptography`.
- `.gitignore`: also ignores `node_modules/`, `.pytest_cache/` and `bench/results/*` raw data. The written-up notes
  in `bench/results/*.md` are tracked because the README links to them.

### Known issues and limits
- **C2PA needs `c2pa-python` 0.37.3 – 0.37.10.** 0.37.12 and 0.38.0 reject the `c2pa.watermarked` action
  (`softBindingMissing`: they want a soft-binding assertion, which this tool does not write yet); 0.37.0 / 0.37.1 raise
  a `TypeError` in the signer. With an unsupported version `--c2pa` reports that the file was not signed.
- **Not measured on any real platform.** Every robustness number comes from local stand-ins (Pillow JPEG, ffmpeg/x264,
  a viewer's resize). Use `bench/repost_roundtrip.py` to measure YouTube, Facebook, Telegram, TikTok, X, Instagram.
- Below about ×0.5 scale the mark itself is gone; a whole-screen screenshot of a small photo at 50% in the corner of a
  black screen does not read.
- Diffusion regeneration / AI repainting: no evidence this stack survives it (the README says so).
- `--ss-key` is a reproducibility value, not a secret.
- The measured corpus is small (about 11 real photos up to ~2 MP, two 10 MP photos, one real video clip); no 12 MP phone
  photos, long video, HEVC or AV1 yet.

### Breaking
- `extract` runs crop resync by default (`--no-resync` restores the old mirrored decodes).
- The `spread` mark is read ID-first, then as text; text payloads still read back as before, but **free-text payloads are
  not recovered from cropped or mirrored copies**, only IDs.
- Nothing from v5 stops working: an image stamped by v5 with `--invisible` (LSB) reads back with the v6 reader
  (checked). v5 outputs simply have no spread mark, ID, ledger entry or C2PA manifest to read.
