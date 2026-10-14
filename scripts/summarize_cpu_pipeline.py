"""Summarize fixed S3 runs; fail on incomplete or inconsistent measurements."""
import argparse
import json
from pathlib import Path
import statistics

SETTINGS = {name: "1" for name in (
    "MGERASE_VIDEO_UINT8_GPU_NORMALIZE", "MGERASE_VIDEO_UINT8_GPU_CLEAR_CACHE", "MGERASE_MASK_PIPE_R_ONLY",
    "MGERASE_STREAMING_COPY_OPT", "MGERASE_MASK_WINDOW_BUFFER_OPT", "MGERASE_OUTPUT_UINT8_GPU_CONVERT",
    "MGERASE_WRITER_HANDOFF_OPT", "MGERASE_MASK_THRESHOLD_INPLACE", "MGERASE_STYLE_UINT8_GPU_UPLOAD",
    "MGERASE_MASK_DILATE_CAST_OPT", "MGERASE_WINDOW_UINT8_VALIDATION_OPT", "MGERASE_TEXT_SESSION_CACHE",
    "MGERASE_MASK_SCAN_REUSE", "MGERASE_FIRST_WINDOW_BORROW",
)}
SETTINGS["MGERASE_PIPE_READ_THREADS"] = "2"


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--reference-sha256", default="")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("runs must be positive")
    reference = args.reference_sha256.lower()
    if reference and (len(reference) != 64 or any(c not in "0123456789abcdef" for c in reference)):
        parser.error("reference-sha256 must contain 64 hexadecimal characters")
    rows, manifests, checks = [], [], {}
    for run in range(1, args.runs + 1):
        directory = args.root / f"run_{run}"
        result = read(directory / "summary.json")
        manifest = read(directory / "manifest.json")
        audit = read(args.root / f"run_{run}-runtime.json")
        sample = result["samples"][0]
        device = manifest["env"]["CUDA_VISIBLE_DEVICES"]
        gpu = result["memory"]["devices"][device]
        warm = [w for w in audit["windows"] if w["is_warmup"]]
        formal = [w for w in audit["windows"] if not w["is_warmup"]]
        checks[f"run_{run}"] = {
            "complete": (args.root / f"run_{run}_COMPLETE").is_file() and result["memory"]["returncode"] == 0 and len(result["samples"]) == 1,
            "all_settings": all(manifest["env"].get(k) == v for k, v in SETTINGS.items()),
            "two_warm_two_formal_windows": len(warm) == len(formal) == 2,
            "cleanup_every_window": bool(audit["windows"]) and all(w["transfer"]["cache_cleanup"].get("applied") is True for w in audit["windows"]),
            "borrow_first_window": all(ws[0]["first_window_borrow"].get("applied") is True and ws[1]["first_window_borrow"].get("applied") is False for ws in (warm, formal)) if len(warm) == len(formal) == 2 else False,
            "formal_session_cache_hit": bool(formal) and formal[0]["text_cache_source"] == "session",
            "memory_target": result["sampled_memory_pass"],
            "frozen_source_unchanged": not read(directory / "source_changes_during_run.json"),
            "145frames": sample["metadata"]["num_frames"] == 145,
            "no_added_diagnostic_sync": not audit["diagnostic_sync_enabled"],
            "deterministic": audit["deterministic_requested"] == "1" and audit["cublas_workspace_config"] == ":4096:8",
            "reference_output": sample["output_sha256"] == reference if reference else True,
        }
        rows.append(dict(run=run, request_seconds=sample["request_seconds"],
                         task_nvml_peak_gib=gpu["task_peak_gib"],
                         cpu_peak_rss_gib=result["memory"]["cpu_peak_rss_sum_gib"],
                         output_sha256=sample["output_sha256"]))
        manifests.append(manifest)
    commands = []
    for manifest in manifests:
        command = list(manifest["command"])
        command[command.index("--task-file") + 1] = "TASK_FILE"
        commands.append(command)
    checks["batch"] = {
        "same_source": all(m["source_sha256"] == manifests[0]["source_sha256"] for m in manifests),
        "same_inputs": all(m["inputs"] == manifests[0]["inputs"] for m in manifests),
        "same_parameters": all(c == commands[0] for c in commands),
        "same_environment": all({k:v for k,v in m["env"].items() if k != "PYTHONPATH"} == {k:v for k,v in manifests[0]["env"].items() if k != "PYTHONPATH"} for m in manifests),
        "same_output": len({r["output_sha256"] for r in rows}) == 1,
    }
    seconds = [row["request_seconds"] for row in rows]
    passed = all(value for group in checks.values() for value in group.values())
    summary = dict(rows=rows, request_seconds=dict(mean=statistics.mean(seconds), min=min(seconds), max=max(seconds),
                  sample_std=statistics.stdev(seconds) if len(seconds) > 1 else None), checks=checks, passed=passed,
                  reference_sha256=reference or None,
                  quality_scope="Compared with supplied reference" if reference else "Repeat outputs compared only; baseline quality not evaluated")
    (args.root / "cpu_pipeline_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit("Verification failed; inspect cpu_pipeline_summary.json before using its mean.")


if __name__ == "__main__":
    main()
