#!/usr/bin/env python3
"""Interleaved resident-session serial / dual GPU CFG comparison; caches off."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch

from entrypoints.cli.erase_eraserdit import _build_parser, _build_server_args, _task_to_sampling_params
from pipelines.session import EraseSession
from utils.determinism import enable_deterministic_mode
from utils.distributed_runtime import destroy_runtime_distributed
from utils.inference_timing import build_ltx095_pure_timing_payload


def main():
    parser = _build_parser()
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if not args.model_path or not args.output_path or args.repeat < 1:
        parser.error("model-path, output-path (JSON), and positive repeat are required")
    if args.cfg_parallel_device is None:
        parser.error("--cfg-parallel-device is required")
    if args.transformer_cache_mode != "off" or args.cache_text_projections:
        parser.error("all caches must be off")
    args.cache_text_projections = False
    os.environ.setdefault("MGERASE_FFMPEG_THREADS", "auto")
    report_path = Path(args.output_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"settings": vars(args), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "determinism": enable_deterministic_mode(), "runs": [], "passed": False}
    server_args = _build_server_args(args)
    session = None
    try:
        started = time.perf_counter()
        session = EraseSession(server_args)
        report["load_seconds"] = time.perf_counter() - started
        for index in range(args.repeat):
            order = ("serial", "parallel") if index % 2 == 0 else ("parallel", "serial")
            for mode in order:
                server_args.pipeline_config.cfg_parallel_device = args.cfg_parallel_device if mode == "parallel" else None
                if args.warmup and index == 0:
                    warmup_params = _task_to_sampling_params({
                        "output": str(report_path.with_suffix(f".{mode}_warmup.mp4")),
                        "num_inference_steps": max(1, args.warmup_steps), "strength": 1.0,
                    }, args)
                    warmup_started = time.perf_counter()
                    session.run(warmup_params)
                    for device in range(torch.cuda.device_count()):
                        torch.cuda.synchronize(device)
                    report.setdefault("warmup_seconds", {})[mode] = time.perf_counter() - warmup_started
                output = report_path.with_suffix(f".{mode}_{index}.mp4")
                params = _task_to_sampling_params({"output": str(output)}, args)
                for device in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                result = session.run(params)
                for device in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                run = {"mode": mode, "repeat": index, "seconds": elapsed, "output": str(output),
                       "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                       "cfg_parallel": result.extra.get("cfg_parallel"),
                       "cache_history": result.extra.get("transformer_cache_history"),
                       "timing": build_ltx095_pure_timing_payload(result.metrics),
                       "peak_allocated_gib": [torch.cuda.max_memory_allocated(d) / 1024**3
                                              for d in range(torch.cuda.device_count())]}
                report["runs"].append(run)
                report_path.write_text(json.dumps(report, indent=2) + "\n")
                if not run["cache_history"] or any(s["mode"] != "off" or s.get("cache_text_projections")
                                                   for s in run["cache_history"]):
                    raise AssertionError("measurement did not confirm all caches disabled")
                if mode == "parallel" and (not run["cfg_parallel"] or not all(
                    s["enabled"] and s["steps"] > 0 for s in run["cfg_parallel"]
                )):
                    raise AssertionError("dual GPU execution was not reported")
                print(f"{mode} repeat={index} seconds={elapsed:.3f} sha256={run['sha256']}", flush=True)
        report["byte_identical"] = len({r["sha256"] for r in report["runs"]}) == 1
        medians = {mode: statistics.median(r["seconds"] for r in report["runs"] if r["mode"] == mode)
                   for mode in ("serial", "parallel")}
        report["median_seconds"] = medians
        report["e2e_speedup"] = medians["serial"] / medians["parallel"]
        report["passed"] = report["byte_identical"]
    finally:
        if session is not None:
            session.close()
        destroy_runtime_distributed()
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit("CFG outputs differ; inspect quality before accepting")


if __name__ == "__main__":
    main()
