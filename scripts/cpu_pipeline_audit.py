"""Observe existing window metadata without adding GPU operations."""
import functools
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))
import entrypoints.cli.erase_eraserdit as cli
import pipelines.runtime.drivers.windowed as driver
import utils.inference_timing as inference_timing
from pipelines.stages.eraserdit_erase.preprocess import EraserDiTErasePreprocessStage


def main():
    windows = []
    coarse = {}
    phase = {"formal": False}
    original_forward = EraserDiTErasePreprocessStage.forward

    @functools.wraps(original_forward)
    def forward(self, batch, server_args):
        result = original_forward(self, batch, server_args)
        windows.append({
            "is_warmup": bool(batch.is_warmup),
            "window_index": batch.extra["window_index"],
            "window_shape": list(batch.video.shape),
            "transfer": dict(batch.extra["video_input_transfer"]),
            "first_window_borrow": dict(batch.extra.get("first_window_borrow", {})),
            "text_cache_source": batch.extra.get("text_encoding_cache_source"),
        })
        return result

    EraserDiTErasePreprocessStage.forward = forward
    original_runtime = driver._run_writer_window_runtime

    @functools.wraps(original_runtime)
    def runtime(*args, **kwargs):
        phase["formal"] = not kwargs["batch"].is_warmup
        try:
            return original_runtime(*args, **kwargs)
        finally:
            phase["formal"] = False

    driver._run_writer_window_runtime = runtime
    original_timer = inference_timing.record_diagnostic_stage

    @functools.wraps(original_timer)
    def timer(metrics, name, duration_s):
        if phase["formal"]:
            coarse[name] = coarse.get(name, 0.0) + duration_s
        return original_timer(metrics, name, duration_s)

    for module in list(sys.modules.values()):
        if module is not None and getattr(module, "__name__", "").startswith(("pipelines.", "utils.")):
            for key, value in list(vars(module).items()):
                if value is original_timer:
                    setattr(module, key, timer)

    cli.main()
    Path(os.environ["ERASERDIT_CPU_PIPELINE_AUDIT"]).write_text(json.dumps({
        "windows": windows,
        "coarse_runtime_seconds": coarse,
        "diagnostic_sync_enabled": os.environ.get("MGERASE_DIAGNOSTIC_TIMING") == "1",
        "deterministic_requested": os.environ.get("ERASERDIT_DETERMINISTIC"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "note": "Existing metadata/wall durations only; no extra CUDA sync, events, GC, cache clear, NVTX or profiler.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
