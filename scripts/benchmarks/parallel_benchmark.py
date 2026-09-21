#!/usr/bin/env python3
"""Resident-session EraserDiT CFG/SP/VAE matrix; independent of cache benchmarks."""
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from entrypoints.cli.erase_eraserdit import _build_parser, _build_server_args, _task_to_sampling_params
from parallel.eraserdit_mesh import resolve_mesh
from pipelines.session import EraseSession
from utils.determinism import enable_deterministic_mode
from utils.distributed_runtime import destroy_runtime_distributed
from utils.inference_timing import build_ltx095_pure_timing_payload


CONFIGS = {
    "spatial_vae2": (1, 1, 2, False), "cfg2_spatial_vae2": (1, 2, 2, False),
    "sp2_spatial_vae2": (2, 1, 2, False), "spatial_vae4": (1, 1, 4, False),
    "cfg2_sp2_spatial_vae4": (2, 2, 4, False), "sp4_spatial_vae4": (4, 1, 4, False),
    "serial": (1, 1, 1, False), "cfg2": (1, 2, 1, False),
    "sp2": (2, 1, 1, False), "tiled": (1, 1, 1, True),
    "vae2": (1, 1, 2, True), "cfg2_vae2": (1, 2, 2, True),
    "sp2_vae2": (2, 1, 2, True), "cfg2_sp2": (2, 2, 1, False),
    "sp4": (4, 1, 1, False), "vae4": (1, 1, 4, True),
    "cfg2_sp2_vae4": (2, 2, 4, True), "sp4_vae4": (4, 1, 4, True),
}


def main():
    parser = _build_parser()
    parser.add_argument("--matrix-configs", nargs="+", choices=CONFIGS, default=["serial", "cfg2", "sp2"])
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if not args.output_path or not args.model_path or args.repeat < 1:
        parser.error("model-path, output-path JSON and positive repeat required")
    if args.cfg_parallel_device or args.parallel_devices:
        parser.error("matrix sets its own local device groups")
    if args.transformer_cache_mode != "off" or args.cache_text_projections:
        parser.error("all transformer caches must be disabled")
    args.cache_text_projections = False
    os.environ.setdefault("MGERASE_FFMPEG_THREADS", "auto")
    path = Path(args.output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"settings": vars(args), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "determinism": enable_deterministic_mode(), "runs": [], "completed": False}
    server_args = _build_server_args(args)
    def configure(name):
        c = server_args.pipeline_config
        c.sp_degree, c.cfg_degree, c.vae_degree, c.vae_tiling = CONFIGS[name]
        resolve_mesh(server_args)
    for name in args.matrix_configs:
        configure(name)
    configure("serial")
    session = None
    try:
        started = time.perf_counter()
        session = EraseSession(server_args)
        report["load_seconds"] = time.perf_counter() - started
        for index in range(args.repeat):
            names = args.matrix_configs[index % len(args.matrix_configs):] + args.matrix_configs[:index % len(args.matrix_configs)]
            for name in names:
                configure(name)
                if args.warmup and index == 0:
                    warmup = _task_to_sampling_params({"output": str(path.with_suffix(f".{name}_warmup.mp4")),
                        "num_inference_steps": max(1, args.warmup_steps), "strength": 1.0}, args)
                    session.run(warmup)
                output = path.with_suffix(f".{name}_{index}.mp4")
                params = _task_to_sampling_params({"output": str(output)}, args)
                for device in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                result = session.run(params)
                for device in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(device)
                run = {"mode": name, "repeat": index, "seconds": time.perf_counter() - started,
                       "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "output": str(output),
                       "parallel_history": result.extra.get("parallel_history", []),
                       "cache_history": result.extra.get("transformer_cache_history", []),
                       "timing": build_ltx095_pure_timing_payload(result.metrics),
                       "peak_allocated_gib": [torch.cuda.max_memory_allocated(d) / 1024**3
                                              for d in range(torch.cuda.device_count())]}
                report["runs"].append(run)
                path.write_text(json.dumps(report, indent=2) + "\n")
                if not run["cache_history"] or any(s["mode"] != "off" or s.get("cache_text_projections")
                                                   for s in run["cache_history"]):
                    raise AssertionError("cache-off execution not confirmed")
                print(f"{name}: {run['seconds']:.3f}s sha256={run['sha256']}", flush=True)
        report["completed"] = True
    finally:
        if session is not None:
            session.close()
        destroy_runtime_distributed()
        path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
