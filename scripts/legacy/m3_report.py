#!/usr/bin/env python3
"""Aggregate the M3 sweep into the plan's report matrix.

    python scripts/legacy/m3_report.py --results-dir results/m3 results/m3-clean

Timings come from the per-repeat JSONL, quality from comparing each config's
first repeat against the reference (`sdpa` is the unaccelerated architecture
N).  SSIM/PSNR are computed with ffmpeg's own filters on the whole frame, the
same yardstick the M1b gates use.

Multiple result directories merge left to right, so a re-measured config
overrides its earlier row; videos come from `--video-dir` (default: the first
directory) since the merged JSONL does not say where a run was written.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
from pathlib import Path

# Plan §6.3: the acceleration rows must clear these against N.
MIN_DENOISE_GAIN = 0.15
MIN_E2E_GAIN = 0.10
MIN_SSIM = 0.95
MIN_PSNR_DB = 28.0
MAX_RESERVED_RATIO = 1.05

PREFLIGHT_RE = re.compile(r"Attention backend preflight: (\{.*\})")


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def read_sweep(results_dirs: list[Path]) -> dict[str, list[dict]]:
    """Merge the result directories; a later directory overrides an earlier one.

    That is what lets `results/m3-clean` replace the configs the main sweep
    measured while a second stream was competing for the host.
    """
    sweep: dict[str, list[dict]] = {}
    for results_dir in results_dirs:
        for path in sorted(results_dir.glob("*.jsonl")):
            sweep[path.stem] = [
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip()
            ]
    return sweep


def effective_backend(results_dirs: list[Path], config: str) -> str | None:
    """The backend `auto` actually resolved to, read off the run log."""
    for results_dir in reversed(results_dirs):
        for log in sorted(results_dir.glob(f"{config}_run*.log")):
            match = PREFLIGHT_RE.search(log.read_text(errors="replace"))
            if match:
                try:
                    return json.loads(match.group(1).replace("'", '"')).get("effective")
                except json.JSONDecodeError:
                    return None
    return None


def ffmpeg_metric(a: Path, b: Path, filt: str) -> float | None:
    result = subprocess.run(
        [
            "ffmpeg", "-v", "info", "-i", str(a), "-i", str(b),
            "-lavfi", f"[0:v][1:v]{filt}", "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    pattern = r"All:([0-9.]+)" if filt == "ssim" else r"average:([0-9.inf]+)"
    match = re.search(pattern, result.stderr)
    if not match:
        return None
    value = match.group(1)
    return float("inf") if value == "inf" else float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", nargs="+", default=["results/m3"])
    parser.add_argument("--reference", default="sdpa")
    parser.add_argument(
        "--video-dir",
        default=None,
        help="directory holding <config>_run1.mp4 (defaults to the first --results-dir)",
    )
    args = parser.parse_args()
    results_dirs = [Path(value) for value in args.results_dir]
    video_dirs = (
        [Path(args.video_dir)] if args.video_dir else list(reversed(results_dirs))
    )

    def find_video(config: str) -> Path | None:
        """First `<config>_run1.mp4` in any result directory, newest dir first."""
        for directory in video_dirs:
            candidate = directory / f"{config}_run1.mp4"
            if candidate.exists():
                return candidate
        return None

    sweep = read_sweep(results_dirs)
    if args.reference not in sweep:
        raise SystemExit(f"reference config {args.reference!r} has no results")

    def stats(config: str) -> dict:
        ok = [r for r in sweep[config] if not r.get("failed")]
        stages = [r.get("stages_ms") or {} for r in ok]
        return {
            "runs": len(ok),
            "failed": len(sweep[config]) - len(ok),
            "e2e": median([r["e2e_seconds"] for r in ok if r.get("e2e_seconds")]),
            "denoise": median(
                [
                    s["EraserDiTEraseDenoisingStage"] / 1000.0
                    for s in stages
                    if "EraserDiTEraseDenoisingStage" in s
                ]
            ),
            "step_med": median(
                [r["step_median_ms"] for r in ok if r.get("step_median_ms")]
            ),
            "reserved": median(
                [r["peak_reserved_gib"] for r in ok if r.get("peak_reserved_gib")]
            ),
            "warmup": median(
                [r["warmup_seconds"] for r in ok if r.get("warmup_seconds")]
            ),
            "backend": effective_backend(results_dirs, config),
        }

    base = stats(args.reference)
    reference_video = find_video(args.reference)
    print(
        f"N = {args.reference}  (effective backend {base['backend']}, "
        f"runs {base['runs']}, failed {base['failed']})\n"
    )
    header = (
        f"{'config':16s} {'backend':12s} {'t_e2e':>8s} {'Δe2e':>7s} "
        f"{'t_den':>8s} {'Δden':>7s} {'step':>7s} {'warm':>6s} {'resvGiB':>8s} "
        f"{'SSIM':>6s} {'PSNR':>7s} {'gate':>6s}"
    )
    print(header)
    print("-" * len(header))

    rows: list[str] = []
    for config in sweep:
        if config == args.reference:
            continue
        row = stats(config)
        e2e_gain = (
            (base["e2e"] - row["e2e"]) / base["e2e"] if base["e2e"] and row["e2e"] else None
        )
        den_gain = (
            (base["denoise"] - row["denoise"]) / base["denoise"]
            if base["denoise"] and row["denoise"]
            else None
        )
        video = find_video(config)
        comparable = video is not None and reference_video is not None
        ssim = ffmpeg_metric(reference_video, video, "ssim") if comparable else None
        psnr = ffmpeg_metric(reference_video, video, "psnr") if comparable else None
        reserved_ok = (
            row["reserved"] <= base["reserved"] * MAX_RESERVED_RATIO
            if row["reserved"] and base["reserved"]
            else None
        )
        passes = (
            den_gain is not None and den_gain >= MIN_DENOISE_GAIN
            and e2e_gain is not None and e2e_gain >= MIN_E2E_GAIN
            and reserved_ok is True
            and ssim is not None and ssim >= MIN_SSIM
            and psnr is not None and psnr >= MIN_PSNR_DB
            and row["failed"] == 0
        )
        rows.append(
            f"{config:16s} {str(row['backend']):12s} "
            f"{row['e2e'] or 0:8.1f} {e2e_gain * 100 if e2e_gain else 0:+6.1f}% "
            f"{row['denoise'] or 0:8.1f} {den_gain * 100 if den_gain else 0:+6.1f}% "
            f"{row['step_med'] or 0:7.0f} {row['warmup'] or 0:6.1f} "
            f"{row['reserved'] or 0:8.2f} "
            f"{ssim or 0:6.4f} {psnr or 0:7.2f} {'PASS' if passes else 'fail':>6s}"
        )
    print("\n".join(rows))
    print(
        f"\ngates: Δdenoise >= {MIN_DENOISE_GAIN:.0%}, Δe2e >= {MIN_E2E_GAIN:.0%}, "
        f"reserved <= N x {MAX_RESERVED_RATIO}, SSIM >= {MIN_SSIM}, "
        f"PSNR >= {MIN_PSNR_DB} dB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
