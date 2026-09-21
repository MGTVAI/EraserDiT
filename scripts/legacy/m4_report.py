#!/usr/bin/env python3
"""Assemble the M4 acceptance matrix (plan §6.1) for both clips.

    python scripts/legacy/m4_report.py [--clip 10268234 ...] [--skip-quality]

Three configurations per clip:

* ``B`` frozen baseline, from ``EraserDiT-baseline/baseline_runner.py``
  (its summary JSON, a different codebase and a different JSON shape);
* ``N`` this architecture, unaccelerated (``--attention-backend sdpa``);
* ``A`` the recommended accelerated configuration (``sage_attn`` +
  ``torch.compile`` + ``--warmup``).

Quality uses the plan's fixed caliber, ``EraserDiT-baseline/compare_outputs.py``
(ffmpeg ``ssim``/``psnr`` filters on the Y plane, encoding noise included), for
both the whole frame and the non-erased region M1b defines.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASELINE = Path("/mnt/shanhai-ai/shanhai-workspace/zhouhao6/EraserDiT-baseline")
COMPARE = BASELINE / "compare_outputs.py"
PYTHON = "/mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python"

# Plan §6.3.
MIN_DENOISE_GAIN = 0.15
MIN_E2E_GAIN = 0.10
MAX_N_OVER_B_E2E = 1.15
MAX_RESERVED_OVER_BASELINE = 1.10
MIN_QUALITY_SSIM = 0.95
MIN_QUALITY_PSNR_DB = 28.0
MIN_NONMASKED_SSIM = 0.99
MIN_NONMASKED_PSNR_DB = 40.0
# The 0.99 / 40 dB equivalence gate above is the M1b gate and it is defined for
# the deterministic profile (plan §6.4: equivalence in the deterministic
# profile, performance in the fast one).  This matrix is fast-profile, where
# the baseline cannot even reproduce *itself* to better than plan §6.2's
# measured band, so that band is the honest bar for N vs B here.
BASELINE_SELF_SSIM = 0.9889
BASELINE_SELF_PSNR_DB = 38.67


def median(values):
    clean = [v for v in values if v is not None]
    return statistics.median(clean) if clean else None


def spread(values) -> str:
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return ""
    return f"±{(max(clean) - min(clean)) / 2:.1f}"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


PREFLIGHT_RE = re.compile(r"Attention backend preflight: (\{.*\})")


def preflight_backend(results_dir: Path, config: str) -> str | None:
    """The backend the request actually resolved to, read off the run log.

    `result.extra["attention_backend"]` is not populated on the CLI path, and
    the plan requires the matrix to name what `auto` selected, so the log line
    the pipeline emits at construction is the source of truth here.
    """
    for log in sorted(results_dir.glob(f"{config}_run*.log")):
        match = PREFLIGHT_RE.search(log.read_text(errors="replace"))
        if match:
            try:
                return json.loads(match.group(1).replace("'", '"')).get("effective")
            except json.JSONDecodeError:
                return None
    return None


def compare(a: Path, b: Path, mask: Path) -> dict:
    """Both calibers from one run of the baseline's own comparator."""
    if not (a.exists() and b.exists()):
        return {}
    result = subprocess.run(
        [PYTHON, str(COMPARE), str(a), str(b), "--mask", str(mask)],
        capture_output=True,
        text=True,
        cwd=BASELINE,
    )
    if result.returncode != 0:
        return {}
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    whole = payload.get("regions", {}).get("whole", {})
    masked = payload.get("pixelwise_nonmasked", {})
    return {
        "whole_ssim": whole.get("ssim_y"),
        "whole_psnr": whole.get("psnr_y_db"),
        "nonmasked_ssim": masked.get("ssim_y_nonmasked"),
        "nonmasked_psnr": masked.get("psnr_y_db_nonmasked"),
    }


def baseline_stats(summary: dict, task_index: int) -> dict:
    runs = [r for r in summary.get("runs", []) if r["task"] == task_index]
    return {
        "e2e": summary.get("median_seconds", {}).get(str(task_index)),
        "e2e_spread": spread([r["seconds"] for r in runs]),
        "peak_allocated": median([r["peak_allocated_gib"] for r in runs]),
        "peak_reserved": median([r["peak_reserved_gib"] for r in runs]),
        "load": summary.get("load_seconds"),
        "output": runs[0]["output"] if runs else None,
        "runs": len(runs),
    }


def native_stats(records: list[dict]) -> dict:
    ok = [r for r in records if not r.get("failed")]
    stages = [r.get("stages_ms") or {} for r in ok]

    def stage(name):
        return median([s[name] / 1000.0 for s in stages if name in s])

    return {
        "e2e": median([r["e2e_seconds"] for r in ok]),
        "e2e_spread": spread([r["e2e_seconds"] for r in ok]),
        "denoise": stage("EraserDiTEraseDenoisingStage"),
        "text": stage("EraserDiTEraseTextEncodingStage"),
        "vae": stage("EraserDiTEraseDecodingStage"),
        "step_med": median([r.get("step_median_ms") for r in ok]),
        "peak_allocated": median([r.get("peak_allocated_gib") for r in ok]),
        "peak_reserved": median([r.get("peak_reserved_gib") for r in ok]),
        "load": median([r.get("load_seconds") for r in ok]),
        "warmup": median([r.get("warmup_seconds") for r in ok]),
        "compile": median(
            [
                (r.get("torch_compile") or {}).get("compile_seconds")
                for r in ok
                if r.get("torch_compile")
            ]
        ),
        "backend": None,  # filled from the run log by the caller
        "io": median(
            [
                sum((r.get("runtime_timing_seconds") or {}).values())
                for r in ok
                if r.get("runtime_timing_seconds")
            ]
        ),
        "runs": len(ok),
        "failed": len(records) - len(ok),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", nargs="+", default=["10268234", "113000356"])
    parser.add_argument("--results-dir", default="results/m4")
    parser.add_argument("--baseline-summary", default=str(BASELINE / "results/m4_baseline_summary.json"))
    parser.add_argument("--skip-quality", action="store_true")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    summary_path = Path(args.baseline_summary)
    baseline = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    for clip_index, clip in enumerate(args.clips):
        print(f"\n{'=' * 78}\n素材 {clip}\n{'=' * 78}")
        native = {
            config: native_stats(load_jsonl(results_dir / f"{clip}_{config}.jsonl"))
            for config in ("N", "A")
        }
        b = baseline_stats(baseline, clip_index) if baseline else {}

        header = (
            f"{'cfg':4s} {'runs':>4s} {'t_load':>7s} {'t_e2e':>8s} {'±':>5s} "
            f"{'t_den':>7s} {'t_step':>7s} {'t_vae':>6s} {'t_text':>7s} {'t_io':>6s} "
            f"{'t_warm':>7s} {'allocGiB':>9s} {'resvGiB':>8s}"
        )
        print(header)
        print("-" * len(header))
        print(
            f"{'B':4s} {b.get('runs', 0):4d} {b.get('load') or 0:7.1f} "
            f"{b.get('e2e') or 0:8.1f} {b.get('e2e_spread', ''):>5s} "
            f"{'-':>7s} {'-':>7s} {'-':>6s} {'-':>7s} {'-':>6s} {'-':>7s} "
            f"{b.get('peak_allocated') or 0:9.2f} {b.get('peak_reserved') or 0:8.2f}"
        )
        for config in ("N", "A"):
            s = native[config]
            if not s["runs"]:
                print(f"{config:4s}    0  (no results)")
                continue
            print(
                f"{config:4s} {s['runs']:4d} {s['load'] or 0:7.1f} "
                f"{s['e2e'] or 0:8.1f} {s['e2e_spread']:>5s} "
                f"{s['denoise'] or 0:7.1f} {s['step_med'] or 0:7.0f} "
                f"{s['vae'] or 0:6.1f} {s['text'] or 0:7.1f} {s['io'] or 0:6.1f} "
                f"{s['warmup'] or 0:7.1f} {s['peak_allocated'] or 0:9.2f} "
                f"{s['peak_reserved'] or 0:8.2f}"
            )

        b_e2e = b.get("e2e")
        n, a = native["N"], native["A"]
        if b_e2e and n["e2e"] and a["e2e"] and n["denoise"] and a["denoise"]:
            print(
                f"\n  t_e2e(N)/t_e2e(B)      = {n['e2e'] / b_e2e:.3f}  "
                f"(线 ≤ {MAX_N_OVER_B_E2E})"
            )
            b_resv = b.get("peak_reserved")
            if b_resv:
                print(
                    f"  peak_reserved(N)/B     = {n['peak_reserved'] / b_resv:.3f}  "
                    f"(线 ≤ {MAX_RESERVED_OVER_BASELINE})"
                )
            print(
                f"  Δdenoise(A vs N)       = {(n['denoise'] - a['denoise']) / n['denoise']:+.1%}  "
                f"(线 ≥ {MIN_DENOISE_GAIN:.0%})"
            )
            print(
                f"  Δe2e(A vs N)           = {(n['e2e'] - a['e2e']) / n['e2e']:+.1%}  "
                f"(线 ≥ {MIN_E2E_GAIN:.0%})"
            )

        if args.skip_quality:
            continue
        # compare_outputs.py runs with cwd=BASELINE, so every path handed to it
        # has to be absolute.
        mask = (REPO / f"data/{clip}_mask.mp4").resolve()
        source = (REPO / f"data/{clip}.mp4").resolve()
        b_video = (BASELINE / (b.get("output") or "")).resolve()
        n_video = (REPO / results_dir / f"{clip}_N_run1.mp4").resolve()
        a_video = (REPO / results_dir / f"{clip}_A_run1.mp4").resolve()
        print("\n  质量（口径：compare_outputs.py，ffmpeg ssim/psnr 滤镜，Y 平面）")
        print(f"    {'pair':16s} {'整帧 SSIM':>10s} {'整帧 PSNR':>10s} "
              f"{'非擦除 SSIM':>12s} {'非擦除 PSNR':>12s}")
        for label, left, right in (
            ("B vs 源", b_video, source),
            ("N vs 源", n_video, source),
            ("A vs 源", a_video, source),
            ("N vs B", n_video, b_video),
            ("A vs N", a_video, n_video),
        ):
            metrics = compare(left, right, mask)
            if not metrics:
                print(f"    {label:16s} (缺文件)")
                continue
            print(
                f"    {label:16s} {metrics.get('whole_ssim') or 0:10.4f} "
                f"{metrics.get('whole_psnr') or 0:10.2f} "
                f"{metrics.get('nonmasked_ssim') or 0:12.4f} "
                f"{metrics.get('nonmasked_psnr') or 0:12.2f}"
            )
        q = compare(a_video, n_video, mask)
        if q:
            ok = (
                (q.get("whole_ssim") or 0) >= MIN_QUALITY_SSIM
                and (q.get("whole_psnr") or 0) >= MIN_QUALITY_PSNR_DB
            )
            print(f"    A vs N 整帧门槛（≥ {MIN_QUALITY_SSIM}/{MIN_QUALITY_PSNR_DB} dB）: "
                  f"{'PASS' if ok else 'FAIL'}")
        eq = compare(n_video, b_video, mask)
        if eq:
            ok = (
                (eq.get("nonmasked_ssim") or 0) >= BASELINE_SELF_SSIM
                and (eq.get("nonmasked_psnr") or 0) >= BASELINE_SELF_PSNR_DB
            )
            print(
                f"    N vs B 落在基线自洽带内"
                f"（非确定性口径 ≥ {BASELINE_SELF_SSIM}/{BASELINE_SELF_PSNR_DB} dB）: "
                f"{'PASS' if ok else 'FAIL'}"
            )
            print(
                f"    N vs B 确定性口径门槛（≥ {MIN_NONMASKED_SSIM}/{MIN_NONMASKED_PSNR_DB} dB）"
                f"：由 M1b 在确定性口径下验收，本表为快速口径，不适用"
            )

    print("\n条件")
    print(f"  baseline summary   {summary_path}  seed={baseline.get('seed')}")
    print(f"  baseline determinism {baseline.get('determinism')}")
    print(f"  gpu                {baseline.get('gpu')}")
    print("  native profile     ERASERDIT_DETERMINISTIC=0（快速口径）")
    for config in ("N", "A"):
        for clip in args.clips:
            backend = preflight_backend(results_dir, f"{clip}_{config}")
            if backend:
                print(f"  {config} backend ({clip})  {backend}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
