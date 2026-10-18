"""Metric primitives for the benchmark: confidence intervals, bit error rate
against a known payload, PSNR, and the per-group aggregation that turns raw
per-trial records (see run_benchmark.py for the schema) into summary rows.

Pure numpy/stdlib on purpose - no scipy - so the benchmark adds no
dependency beyond what requirements.txt already pulls in.

Terminology used throughout (and repeated in bench/README.md):

  outcome        one of "correct" (decoder returned exactly the embedded
                 text), "wrong" (returned some other text) or "none"
                 (returned nothing).
  detection rate correct / n over MARKED, ATTACKED images.
  misdecode rate wrong / n over the same images: the decoder confidently
                 returned garbage. Reported separately from a plain miss
                 because it is the more dangerous failure.
  FPR            (correct + wrong) / n over images that carry NO mark for the
                 key being tried (never stamped, or stamped under a different
                 key): *any* returned text counts, because a decoder that
                 says "found: xyz" on an unmarked image is wrong whatever xyz is.
  BER (raw)      per-block / per-pixel hard-decision error against the true
                 tiled bit sequence, BEFORE combining repeats. 0.5 = pure
                 noise. This is the channel quality.
  BER (combined) error of the per-bit decision AFTER summing every repeat at
                 the true period/phase - i.e. what spread.py's "processing
                 gain" buys. Both are measured with oracle alignment (the
                 benchmark knows the geometry of the attack it applied), so
                 they say how much SIGNAL survived independent of whether the
                 decoder managed to re-synchronise. Raw BER can be double
                 digit while combined BER is 0.
"""

import math

import numpy as np

# Decoders that run the --rescale search (mirrors bench_schemes.RESCALE_DECODERS; this module stays free of wmcore).
RESCALE_DECODERS = ("ss_rescale", "ss_id_rescale", "ss_or_id_rescale")

Z95 = 1.959963984540054


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

def wilson_interval(k, n, z=Z95):
    """Wilson score interval for a binomial proportion. Better behaved than
    the normal approximation near 0 and 1, which is where detection rates
    and FPRs live."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _binom_cdf(k, n, p):
    """P(X <= k) for X ~ Binomial(n, p), summed in log space."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    lgn = math.lgamma(n + 1)
    total = 0.0
    for i in range(k + 1):
        total += math.exp(lgn - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
    return min(1.0, total)


def clopper_pearson(k, n, alpha=0.05):
    """Exact (Clopper-Pearson) two-sided interval, by bisection on the
    binomial CDF. Used for FPR, where k is often 0 and the interval's upper
    end - the largest FPR still consistent with what was observed - is the
    number that matters: 0 false positives in 30 images is compatible with a
    true FPR of ~11%, which is a very different claim from "FPR = 0"."""
    if n <= 0:
        return (0.0, 1.0)
    half = alpha / 2.0
    if k <= 0:
        lo = 0.0
    else:  # smallest p with P(X >= k | p) >= alpha/2
        a, b = 0.0, k / n
        for _ in range(80):
            m = (a + b) / 2.0
            if 1.0 - _binom_cdf(k - 1, n, m) < half:
                a = m
            else:
                b = m
        lo = (a + b) / 2.0
    if k >= n:
        hi = 1.0
    else:  # largest p with P(X <= k | p) >= alpha/2
        a, b = k / n, 1.0
        for _ in range(80):
            m = (a + b) / 2.0
            if _binom_cdf(k, n, m) > half:
                a = m
            else:
                b = m
        hi = (a + b) / 2.0
    return (lo, hi)


# ---------------------------------------------------------------------------
# BER / PSNR
# ---------------------------------------------------------------------------

def ber_raw(soft, orig_idx, bits):
    """Fraction of observations whose hard decision (soft > 0) disagrees with
    the true bit at that observation's ORIGINAL raster position.

    `soft`      one float per observation (block correlation for spread-
                spectrum, +-1 for an LSB) - positive means "bit = 1".
    `orig_idx`  the index each observation had in the ORIGINAL, unattacked
                image's tiling (see bench_schemes for how a crop/resize is
                mapped back). The payload is tiled periodically, so the true
                bit is bits[orig_idx % period].
    `bits`      the embedded header+payload bit list (wmcore.text_to_bits).
    """
    bits = np.asarray(bits, dtype=np.uint8)
    truth = bits[np.asarray(orig_idx) % len(bits)]
    return float(np.mean((np.asarray(soft) > 0).astype(np.uint8) != truth))


def ber_combined(soft, orig_idx, bits):
    """Per-bit error AFTER summing `soft` over every repeat at the true
    period/phase (the soft combining find_spread_spectrum_payload performs,
    but with the phase given instead of searched for)."""
    bits = np.asarray(bits, dtype=np.uint8)
    period = len(bits)
    combined = np.bincount(np.asarray(orig_idx) % period, weights=np.asarray(soft, dtype=np.float64),
                           minlength=period)
    return float(np.mean((combined > 0).astype(np.uint8) != bits))


def psnr(a_rgb, b_rgb):
    """PSNR in dB between two uint8 HxWx3 arrays. A guard rail rather than a
    perceptual metric: sweeping --ss-alpha makes detection look monotonically
    better, so every sweep needs the imperceptibility cost next to it."""
    a = np.asarray(a_rgb, dtype=np.float64)
    b = np.asarray(b_rgb, dtype=np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(255.0 ** 2 / mse)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

GROUP_FIELDS = ("kind", "decoder", "attack", "family", "gate", "image_kind", "size", "alpha", "payload_id")


def group_key(rec):
    return tuple(rec.get(f) if not isinstance(rec.get(f), list) else tuple(rec.get(f)) for f in GROUP_FIELDS)


def _pct(values, q):
    values = [v for v in values if v is not None]
    return float(np.percentile(values, q)) if values else None


def _mean(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(values)) if values else None


def summarize_group(recs):
    """One summary row for a list of trial records sharing a group key."""
    n = len(recs)
    correct = sum(1 for r in recs if r["outcome"] == "correct")
    wrong = sum(1 for r in recs if r["outcome"] == "wrong")
    none = n - correct - wrong
    first = recs[0]
    row = {f: first.get(f) for f in GROUP_FIELDS}
    row.update(n=n, correct=correct, wrong=wrong, none=none)
    if first["kind"] == "pos":
        row["det_rate"] = correct / n
        row["det_ci"] = list(wilson_interval(correct, n))
        row["misdecode_rate"] = wrong / n
    else:  # negatives: any returned text is a false positive
        fp = correct + wrong
        row["fp"] = fp
        row["fpr"] = fp / n
        row["fpr_ci"] = list(clopper_pearson(fp, n))
        row["leak"] = correct  # wrong-key only: payload actually recovered
    row["ber_raw_mean"] = _mean([r.get("ber_raw") for r in recs])
    row["ber_comb_mean"] = _mean([r.get("ber_comb") for r in recs])
    secs = [r["decode_s"] for r in recs]
    row["decode_s_p50"] = _pct(secs, 50)
    row["decode_s_p95"] = _pct(secs, 95)
    mp = [r["decode_s"] / r["mp"] for r in recs if r.get("mp")]
    row["decode_s_per_mp_p50"] = _pct(mp, 50)
    row["confidence_mean"] = _mean([r.get("confidence") for r in recs if r["outcome"] != "none"])
    row["budget_exhausted"] = sum(1 for r in recs if r.get("budget_exhausted"))
    row["factor_rel_err_median"] = _pct([r.get("factor_rel_err") for r in recs], 50)
    row["psnr_mean"] = _mean([r.get("psnr") for r in recs])
    row["embed_s_p50"] = _pct([r.get("embed_s") for r in recs], 50)
    return row


def summarize(records):
    groups = {}
    for r in records:
        if r["kind"] == "embed":
            continue
        groups.setdefault(group_key(r), []).append(r)
    return [summarize_group(g) for _, g in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]])]


def runtime_table(records):
    """Wall-clock by operation and image size: the numbers to quote for "how
    long does this take", separate from any attack's pass/fail. Rows:

      embed:<scheme>           embedding one image.
      decode:<decoder>         decoding an unattacked image (for ss_rescale: the
                               fast path, no resize search runs).
      rescale search, ...      the --rescale search proper, split by what it
                               found - the README says it costs "more when
                               nothing is found"; this is where that is measured:
                               "mark found" (resized marked image, recovered),
                               "mark missed" (resized marked image, not recovered)
                               and "unmarked" (nothing to find - the full sweep).
    """
    rows = {}

    def add(label, size, secs, mp, budget=False):
        rows.setdefault((label, tuple(size)), []).append((secs, mp, bool(budget)))

    for r in records:
        if r["kind"] == "embed":
            add("embed:" + r["scheme"], r["size"], r["embed_s"], r["mp"])
        elif r["kind"] == "pos" and r["family"] == "identity":
            add("decode:" + r["decoder"], r["size"], r["decode_s"], r["mp"])
        elif r["kind"] == "pos" and r["decoder"] in RESCALE_DECODERS and r["family"] in ("resize", "resize_jpeg"):
            what = "rescale search" + ("" if r["decoder"] == "ss_rescale" else f" [{r['decoder']}]")
            label = f"{what}, mark found" if r["outcome"] == "correct" else f"{what}, mark missed"
            add(label, r["size"], r["decode_s"], r["mp"], r.get("budget_exhausted"))
        elif r["kind"] == "neg_unmarked" and r["decoder"] in RESCALE_DECODERS:
            what = "rescale search" + ("" if r["decoder"] == "ss_rescale" else f" [{r['decoder']}]")
            add(f"{what}, unmarked (nothing to find)", r["size"], r["decode_s"], r["mp"], r.get("budget_exhausted"))
    out = []
    for (label, size), vals in sorted(rows.items()):
        secs = [v[0] for v in vals]
        out.append({
            "op": label, "size": list(size), "n": len(vals),
            "p50_s": _pct(secs, 50), "p95_s": _pct(secs, 95), "max_s": float(max(secs)),
            "s_per_mp_p50": _pct([s / mp for s, mp, _ in vals if mp], 50),
            "budget_hit": sum(1 for v in vals if v[2]),
        })
    return out


# ---------------------------------------------------------------------------
# Baseline comparison (regression gate)
# ---------------------------------------------------------------------------

def compare_summaries(baseline_rows, current_rows, det_tol=0.05, fpr_tol=0.02, ber_tol=0.02,
                      runtime_factor=None):
    """Compare two summary row lists; returns a list of human-readable
    regression strings (empty = no regression).

    Only rows with gate=True take part in the detection/BER checks: attacks
    the README documents as unsupported (unaligned crop, rotation, ...) are
    reported but never fail a run. A detection-rate drop is flagged when it
    exceeds `det_tol` in absolute terms; because the corpus is seeded, an
    unchanged codebase on the same library versions reproduces the baseline
    exactly, so any drop is a behaviour change, not sampling noise. Runtime
    is only checked when `runtime_factor` is given (machines differ - never
    compare timings across hardware).
    """
    def index(rows):
        return {tuple(tuple(r.get(f)) if isinstance(r.get(f), list) else r.get(f) for f in
                      ("kind", "decoder", "attack", "image_kind", "size", "alpha", "payload_id")): r for r in rows}

    base, cur = index(baseline_rows), index(current_rows)
    problems = []
    for key, c in cur.items():
        b = base.get(key)
        if b is None:
            continue
        size = f"{key[4][0]}x{key[4][1]}" if isinstance(key[4], tuple) and len(key[4]) == 2 else key[4]
        label = f"{key[1]}/{key[2]} [{key[3]} {size} a={key[5]} payload={key[6]}]"
        if c["kind"] == "pos":
            if c.get("gate") and c["det_rate"] < b["det_rate"] - det_tol:
                problems.append(f"detection {label}: {b['det_rate']:.1%} -> {c['det_rate']:.1%}")
            if (c.get("gate") and c["ber_comb_mean"] is not None and b["ber_comb_mean"] is not None
                    and c["ber_comb_mean"] > b["ber_comb_mean"] + ber_tol):
                problems.append(f"combined BER {label}: {b['ber_comb_mean']:.3f} -> {c['ber_comb_mean']:.3f}")
        else:
            if c["fpr"] > b["fpr"] + fpr_tol:
                problems.append(f"FPR {label}: {b['fpr']:.1%} -> {c['fpr']:.1%}")
        if runtime_factor and b.get("decode_s_p50") and c.get("decode_s_p50"):
            if c["decode_s_p50"] > runtime_factor * b["decode_s_p50"]:
                problems.append(f"runtime {label}: p50 {b['decode_s_p50']*1e3:.1f}ms -> {c['decode_s_p50']*1e3:.1f}ms")
    return problems
