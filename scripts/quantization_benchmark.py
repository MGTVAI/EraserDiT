#!/usr/bin/env python3
"""Single-GPU resident BF16 -> INT8 comparison, with real backend counters."""
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from entrypoints.cli.erase_eraserdit import _build_parser, _build_server_args, _task_to_sampling_params
from models.dits.eraserdit_quantization import quantize_transformer, runtime_report, validate_quantization
from pipelines.session import EraseSession
from utils.determinism import enable_deterministic_mode
from utils.distributed_runtime import destroy_runtime_distributed
from utils.inference_timing import build_ltx095_pure_timing_payload





def main():
    parser = _build_parser()
    parser.add_argument("--quant-only", action="store_true")

    args = parser.parse_args()
    if not args.output_path or not args.model_path:
        parser.error("model-path and output-path JSON required")
    if args.cfg_parallel_device or args.parallel_devices or max(args.sp_degree,args.cfg_degree,args.vae_degree)>1 or torch.cuda.device_count()!=1:
        parser.error("quantization benchmark requires exactly one visible GPU and no parallelism")
    if args.transformer_cache_mode != "off" or args.cache_text_projections:
        parser.error("all transformer caches must be disabled")
    args.cache_text_projections = False
    os.environ.setdefault("MGERASE_FFMPEG_THREADS", "auto")
    path = Path(args.output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"settings": vars(args), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "determinism": enable_deterministic_mode(), "runs": [], "completed": False}
    server_args = _build_server_args(args)
    server_args.transformer_quantization = "none"
    modes = ["int8"] if args.quant_only else ["bf16", "int8"]
    session = None
    try:
        started = time.perf_counter()
        session = EraseSession(server_args)
        report["load_seconds"] = time.perf_counter() - started
        for index in range(1):
            for name in modes:
                if name == "int8":
                    server_args.transformer_quantization = "int8_w8a8_native"
                    validate_quantization(server_args)
                    conversion = quantize_transformer(session.pipeline.get_module("transformer"), args.quantization_scope)
                    server_args.effective_transformer_quantization = conversion["mode"]
                    server_args.transformer_quantization_report = conversion
                    torch.cuda.empty_cache()
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
                       "quantization": runtime_report(session.pipeline.get_module("transformer")),
                       "cache_history": result.extra.get("transformer_cache_history", []),
                       "timing": build_ltx095_pure_timing_payload(result.metrics),
                       "peak_allocated_gib": [torch.cuda.max_memory_allocated(d) / 1024**3
                                              for d in range(torch.cuda.device_count())]}
                if name == "int8" and run["quantization"]["executed_module_count"] != run["quantization"]["quantized_count"]:
                    raise AssertionError("not all selected INT8 modules executed")
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
