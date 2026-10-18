"""Render summary rows (bench_metrics.summarize) as Markdown and CSV."""

import csv

import bench_metrics as M


def _pct(x):
    return "n/a" if x is None else f"{x:.1%}"


def _dur(s):
    if s is None:
        return "n/a"
    return f"{s * 1e3:.0f} ms" if s < 1.0 else f"{s:.1f} s"


def _ber(x):
    return "n/a" if x is None else f"{x:.3f}"


def _md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def diagnose(row):
    """Heuristic reading of a failing positive row, using oracle BER to tell
    "the mark is gone" from "the mark is there but the decoder can't align to
    it" - the distinction detection rate alone cannot make."""
    if row["det_rate"] >= 0.95:
        return ""
    b = row["ber_comb_mean"]
    if b is None:
        return "no oracle geometry"
    if b < 0.02:
        return "signal intact - decoder loses sync"
    if b < 0.25:
        return "signal degraded"
    return "signal destroyed"


def headline(rows):
    """The four requested numbers, aggregated, at the top of the report."""
    lines = []
    for dec in ("ss", "ss_id", "ss_flip", "ss_id_flip", "ss_rescale", "ss_id_rescale", "ss_or_id_rescale", "lsb"):
        pos = [r for r in rows if r["kind"] == "pos" and r["decoder"] == dec and r["gate"]]
        if not pos:
            continue
        worst = min(pos, key=lambda r: r["det_rate"])
        n_tot = sum(r["n"] for r in pos)
        ok_tot = sum(r["correct"] for r in pos)
        lo, hi = M.wilson_interval(ok_tot, n_tot)
        lines.append(f"- **`{dec}` detection** over gated attacks: {ok_tot}/{n_tot} = {ok_tot / n_tot:.1%} "
                     f"[{lo:.1%}-{hi:.1%}]; weakest gated attack `{worst['attack']}` at {worst['det_rate']:.1%} (n={worst['n']}).")
    for kind, label in (("neg_unmarked", "unmarked-image FPR"), ("neg_wrongkey", "wrong-key acceptance")):
        for dec in ("lsb", "ss", "ss_id", "ss_flip", "ss_id_flip", "ss_rescale", "ss_id_rescale", "ss_or_id_rescale"):
            neg = [r for r in rows if r["kind"] == kind and r["decoder"] == dec]
            if not neg:
                continue
            n, fp = sum(r["n"] for r in neg), sum(r["fp"] for r in neg)
            lo, hi = M.clopper_pearson(fp, n)
            extra = ""
            if kind == "neg_wrongkey":
                extra = f"; exact payload recovered in {sum(r['leak'] for r in neg)}/{n}"
            lines.append(f"- **`{dec}` {label}**: {fp}/{n} = {fp / n:.1%} (95% upper bound {hi:.1%}){extra}.")
    for rdec in ("ss_rescale", "ss_id_rescale", "ss_or_id_rescale"):
        rt = [r for r in rows if r["kind"] == "pos" and r["decoder"] == rdec and r["attack"] != "identity"]
        if rt:
            hits = [r["decode_s_p50"] for r in rt if r["det_rate"] > 0.5 and r["decode_s_p50"]]
            if hits:
                lines.append(f"- **`{rdec}` runtime** on resized images it can recover: median of per-attack p50 = {_dur(sorted(hits)[len(hits) // 2])}.")
    return "\n".join(lines)


def _pos_section(rows):
    cells = {}
    for r in rows:
        if r["kind"] == "pos":
            cells.setdefault((r["decoder"], r["image_kind"], tuple(r["size"]), r["alpha"], r["payload_id"]), []).append(r)
    parts = []
    for (dec, kind, size, alpha, pid), rs in sorted(cells.items(), key=lambda kv: [str(x) for x in kv[0]]):
        has_rescale = dec in ("ss_rescale", "ss_id_rescale", "ss_or_id_rescale")
        head = ["attack", "n", "detect [95% CI]", "misdecode", "BER raw", "BER comb", "p50", "p95"]
        if has_rescale:
            head += ["confidence", "scale err", "budget hit"]
        head += ["reading"]
        body = []
        psnr = [r["psnr_mean"] for r in rs if r.get("psnr_mean") is not None]
        for r in rs:
            row = [("†" if not r["gate"] else "") + r["attack"], r["n"],
                   f"{r['det_rate']:.1%} [{r['det_ci'][0]:.0%}–{r['det_ci'][1]:.0%}]",
                   _pct(r["misdecode_rate"]), _ber(r["ber_raw_mean"]), _ber(r["ber_comb_mean"]),
                   _dur(r["decode_s_p50"]),
                   _dur(r["decode_s_p95"])]
            if has_rescale:
                row += ["n/a" if r["confidence_mean"] is None else f"{r['confidence_mean']:.2f}",
                        "n/a" if r["factor_rel_err_median"] is None else f"{r['factor_rel_err_median']:.4f}",
                        r["budget_exhausted"] or ""]
            row += [diagnose(r)]
            body.append(row)
        title = f"#### `{dec}` · {kind} {size[0]}×{size[1]}" + (f" · α={alpha:g}" if alpha is not None else "") + f" · payload `{pid}`"
        if psnr:
            title += f" · PSNR {sum(psnr) / len(psnr):.1f} dB"
        parts.append(title + "\n\n" + _md_table(head, body))
    return "\n\n".join(parts)


def _neg_section(rows):
    body = []
    labels = {"neg_unmarked": "unmarked image", "neg_wrongkey": "marked, wrong key"}
    for r in rows:
        if r["kind"] in labels:
            body.append([labels[r["kind"]], r["decoder"], r["attack"], f"{r['image_kind']} {r['size'][0]}×{r['size'][1]}",
                         r["n"], r["fp"], f"{r['fpr']:.1%} (≤{r['fpr_ci'][1]:.1%})",
                         r["leak"] if r["kind"] == "neg_wrongkey" else "—"])
    if not body:
        return ""
    return _md_table(["condition", "decoder", "attack", "images", "n", "FP", "FPR (95% upper)", "payload recovered"], body)


def _runtime_section(runtime):
    if not runtime:
        return ""
    body = [[r["op"], f"{r['size'][0]}×{r['size'][1]}", r["n"], _dur(r["p50_s"]), _dur(r["p95_s"]), _dur(r["max_s"]),
             "n/a" if r["s_per_mp_p50"] is None else _dur(r["s_per_mp_p50"]), r["budget_hit"] or ""] for r in runtime]
    return _md_table(["operation", "size", "n", "p50", "p95", "max", "p50 per MP", "budget hit"], body)


def _alpha_sweep(rows):
    alphas = sorted({r["alpha"] for r in rows if r["kind"] == "pos" and r["decoder"] == "ss" and r["alpha"] is not None})
    if len(alphas) < 2:
        return ""
    pick = [a for a in ("identity", "jpeg_75", "jpeg_50", "noise_5", "blur_1.0") if any(r["attack"] == a for r in rows)]
    cells = sorted({(r["image_kind"], tuple(r["size"]), r["payload_id"]) for r in rows if r["kind"] == "pos" and r["decoder"] == "ss"})
    parts = []
    for kind, size, pid in cells:
        body = []
        for a in alphas:
            sel = {r["attack"]: r for r in rows if r["kind"] == "pos" and r["decoder"] == "ss" and r["alpha"] == a
                   and r["image_kind"] == kind and tuple(r["size"]) == size and r["payload_id"] == pid}
            if not sel:
                continue
            psnr = next((r["psnr_mean"] for r in sel.values() if r["psnr_mean"] is not None), None)
            body.append([f"{a:g}", "n/a" if psnr is None else f"{psnr:.1f}"] +
                        [_pct(sel[p]["det_rate"]) if p in sel else "—" for p in pick])
        parts.append(f"`{kind}` {size[0]}×{size[1]} · payload `{pid}`\n\n" + _md_table(["α", "PSNR dB"] + pick, body))
    return "\n\n".join(parts)


def render_markdown(meta, rows, runtime):
    env, cfg = meta["env"], meta["config"]
    out = [f"# Watermark benchmark — `{meta['suite']}` suite",
           f"_{meta['created']} · {meta['records']} trials in {meta['wall_seconds']} s · seed {meta['seed']}_",
           "",
           f"wmcore fingerprint `{meta['wmcore_fingerprint']}` · Python {env['python']} · numpy {env['numpy']} · "
           f"OpenCV {env['opencv']} · Pillow {env['pillow']} · {env['machine']}, {env['cpu_count']} CPU(s) · jobs={meta['jobs']}"]
    if meta["jobs"] > 1:
        out.append("\n> **Runtime figures below are inflated** (jobs > 1: workers share cores). Re-run with `--jobs 1` before quoting timings.")
    src = f"real images from `{cfg['images_dir']}`" if cfg.get("images_dir") else \
        f"synthetic images ({', '.join(cfg['kinds'])}) — a stand-in for photos; use `--images DIR` for numbers you intend to publish"
    out += ["", f"Corpus: {src}."]
    out += ["", "## Headline", "", headline(rows) or "_(no gated results)_"]
    out += ["", "## Detection and BER  (marked → attacked → decoded)", "",
            "`detect` = exact payload recovered. `misdecode` = a *different* text returned. `BER raw` / `BER comb` use oracle "
            "alignment (see bench/README.md): they measure surviving signal, not decoder success. † = not gated (a documented limit, or outside that decoder's design scope), "
            "still measured. `reading` is a heuristic for rows below 95%.", "", _pos_section(rows)]
    neg = _neg_section(rows)
    if neg:
        out += ["", "## False positives", "",
                "Any text returned counts. With n images the smallest FPR that can be claimed is about 3/n "
                "(rule of three) — the upper bound, not the point estimate, is the honest figure.", "", neg]
    out += ["", "## Runtime", "", _runtime_section(runtime)]
    sweep = _alpha_sweep(rows)
    if sweep:
        out += ["", "## Strength sweep (detection vs α, with the imperceptibility cost)", "", sweep]
    return "\n".join(out) + "\n"


def write_csv(rows, path):
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("x".join(map(str, v)) if k == "size" else
                            (";".join(f"{x:.4f}" for x in v) if isinstance(v, list) else v)) for k, v in r.items()})
