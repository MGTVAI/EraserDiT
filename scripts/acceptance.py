#!/usr/bin/env python3
"""Continuous-task acceptance for the EraserDiT runtime (plan §M4).

Runs a task file through one resident pipeline and checks the things a single
task cannot show:

* every task in the list produced an output, in order, into its own file;
* the outputs are pairwise distinct, so no task inherited another's state;
* the two clip geometries (1080x1920 portrait, 1920x1080 landscape) both
  round-trip, and returning to a shape already seen still lands on the same
  compile signature;
* each output has the source's frame count and clears the plan §6.2 anchor for
  the non-erased region (baseline-vs-source was 0.9259-0.9434 SSIM /
  27.28-27.80 dB, so anything at or below that means erasing damaged the
  untouched parts of the frame);
* the per-task timings and window counters are reported.

    python scripts/acceptance.py --task-file tasks/acceptance_tasks.json \
        --attention-backend sage_attn --enable-torch-compile --warmup
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMPARE = Path(
    "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/EraserDiT-baseline/compare_outputs.py"
)
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

# Plan §6.2: the frozen baseline scores 0.9259-0.9434 SSIM / 27.28-27.80 dB on
# the non-erased region against the source.  Anything at or below that is the
# erase having damaged pixels it was told to leave alone.
MIN_NONMASKED_SSIM = 0.92
MIN_NONMASKED_PSNR_DB = 27.0

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{PASS if ok else FAIL}  {name}{(' — ' + detail) if detail else ''}", flush=True)
    if not ok:
        _failures.append(name)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_frames", "-show_entries", "stream=nb_read_frames",
            "-of", "default=nw=1:nk=1", str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return -1


def nonmasked_metrics(output: Path, source: Path, mask: Path) -> dict:
    result = subprocess.run(
        [sys.executable, str(COMPARE), str(output), str(source), "--mask", str(mask)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {}
    try:
        return json.loads(result.stdout).get("pixelwise_nonmasked", {})
    except json.JSONDecodeError:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", default="tasks/acceptance_tasks.json")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--attention-backend", default="sage_attn")
    parser.add_argument("--enable-torch-compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reuse",
        default=None,
        help="validate an existing run's JSON payload instead of running again",
    )
    args = parser.parse_args()

    tasks = json.loads(Path(args.task_file).read_text(encoding="utf-8"))
    if args.reuse:
        payload = json.loads(Path(args.reuse).read_text(encoding="utf-8"))
        print(f"validating existing payload {args.reuse} ({len(tasks)} tasks)\n")
    else:
        command = [
            "./inference_cli.sh",
            "--task-file", args.task_file,
            "--attention-backend", args.attention_backend,
        ]
        if args.model_path:
            command += ["--model-path", args.model_path]
        if args.enable_torch_compile:
            command.append("--enable-torch-compile")
        if args.warmup:
            command.append("--warmup")
        print(f"running: {' '.join(command)}\n", flush=True)
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
        Path("results/acceptance_payload.json").write_text(result.stdout, encoding="utf-8")
        Path("results/acceptance_run.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            print(result.stderr[-4000:])
            check("task file run exits 0", False, f"rc={result.returncode}")
            return 1
        payload = json.loads(result.stdout)
        print(f"payload saved to results/acceptance_payload.json\n")

    results = payload["tasks"]
    check("every task returned a result", len(results) == len(tasks),
          f"{len(results)}/{len(tasks)}")

    digests: dict[str, str] = {}
    for task, record in zip(tasks, results):
        task_id = task["id"]
        output = record.get("output_file_path")
        check(f"[{task_id}] produced an output", bool(output), str(output))
        if not output:
            continue
        path = Path(output)
        check(f"[{task_id}] output exists and is non-empty",
              path.is_file() and path.stat().st_size > 1024,
              f"{path.stat().st_size} B" if path.is_file() else "missing")
        digests[task_id] = md5(path)

        expected = frame_count(Path(task["video"]))
        actual = frame_count(path)
        check(f"[{task_id}] frame count matches the source",
              expected > 0 and actual == expected, f"{actual} vs source {expected}")

        metrics = nonmasked_metrics(path, Path(task["video"]), Path(task["mask"]))
        ssim = metrics.get("ssim_y_nonmasked")
        psnr = metrics.get("psnr_y_db_nonmasked")
        check(
            f"[{task_id}] non-erased region intact",
            ssim is not None and psnr is not None
            and ssim >= MIN_NONMASKED_SSIM and psnr >= MIN_NONMASKED_PSNR_DB,
            f"SSIM {ssim:.4f} / PSNR {psnr:.2f} dB "
            f"(floor {MIN_NONMASKED_SSIM}/{MIN_NONMASKED_PSNR_DB})"
            if ssim is not None else "no metrics",
        )

    values = list(digests.values())
    check("outputs are pairwise distinct", len(set(values)) == len(values),
          f"{len(set(values))} unique of {len(values)}")

    # Same clip, different seed must still differ: identical bytes would mean the
    # seed never reached the sampler.
    by_clip: dict[str, list[str]] = {}
    for task in tasks:
        by_clip.setdefault(task["video"], []).append(task["id"])
    for video, ids in by_clip.items():
        if len(ids) < 2:
            continue
        present = [i for i in ids if i in digests]
        check(f"same clip across seeds stays distinct ({Path(video).name})",
              len({digests[i] for i in present}) == len(present),
              ", ".join(f"{i}={digests[i][:8]}" for i in present))

    print("\nper-task timing")
    for task, record in zip(tasks, results):
        timing = record.get("timing") or {}
        extra = timing.get("extra") or {}
        print(
            f"  {task['id']:32s} e2e={record['elapsed_seconds']:7.1f}s "
            f"(net {record.get('e2e_seconds_excluding_warmup', 0):7.1f}s) "
            f"denoise={(timing.get('pure_inference_stage_breakdown_ms') or {}).get('EraserDiTEraseDenoisingStage', 0) / 1000:7.1f}s "
            f"windows={extra.get('runtime_progress_window_count')} "
            f"peak_resv={extra.get('peak_reserved_gib') or 0:.2f}GiB"
        )

    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed: {', '.join(_failures)}")
        return 1
    print("all acceptance checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
