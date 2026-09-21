"""Partition independent tasks across disjoint resident EraserDiT GPU workers.

Use CUDA_VISIBLE_DEVICES to explicitly authorize the physical GPU pool. Each
worker can itself use CFG/SP and VAE parallelism; DP never splits video windows.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from entrypoints.cli.erase_eraserdit import _build_parser, _load_tasks, _task_to_sampling_params


def partition_tasks(tasks, degree):
    if degree < 1 or degree > len(tasks):
        raise ValueError("dp-degree must be positive and no larger than the task count")
    return [tasks[i::degree] for i in range(degree)]


def main():
    parser = _build_parser()
    parser.add_argument("--dp-degree", type=int, required=True)
    parser.add_argument("--parallel-run-dir", type=Path, required=True)
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if not all(visible) or len(set(visible)) != len(visible):
        parser.error("set an explicit, unique CUDA_VISIBLE_DEVICES pool")
    if args.parallel_devices is not None or args.device not in ("cuda", "cuda:0"):
        parser.error("DP assigns worker-local devices; use default --device and --parallel-devices")
    size = max(args.sp_degree * args.cfg_degree, args.vae_degree, 2 if args.cfg_parallel_device else 1)
    if args.dp_degree < 1 or len(visible) != args.dp_degree * size:
        parser.error("visible GPU count must equal dp_degree * max(sp_degree*cfg_degree, vae_degree)")
    tasks = _load_tasks(args)
    paths = []
    for index, task in enumerate(tasks):
        task.setdefault("id", f"task{index:03d}")
        params = _task_to_sampling_params(task, args)
        if params.transformer_cache_mode != "off" or params.cache_text_projections:
            parser.error("parallel workers require all transformer caches disabled")
        paths.append(str(Path(params.output_path, params.output_file_name).resolve()))
    if len(set(paths)) != len(paths):
        parser.error("every task must have a distinct output path")
    groups = partition_tasks(tasks, args.dp_degree)
    directory = args.parallel_run_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    # Strip dispatcher-only options; preserve all inference options literally.
    filtered, skip = [], False
    removed = {"--dp-degree", "--parallel-run-dir", "--task-file"}
    for arg in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if arg in removed:
            skip = True
        elif arg.split("=", 1)[0] not in removed:
            filtered.append(arg)
    workers, logs = [], []
    report = {"cuda_visible_devices": visible, "dp_degree": args.dp_degree,
              "worker_size": size, "workers": [], "passed": False}
    started = time.perf_counter()
    try:
        for rank, group in enumerate(groups):
            task_file = directory / f"worker{rank}.tasks.json"
            task_file.write_text(json.dumps(group, indent=2) + "\n")
            devices = visible[rank * size:(rank + 1) * size]
            command = [sys.executable, "-m", "entrypoints.cli.erase_eraserdit", *filtered,
                       "--task-file", str(task_file)]
            log = (directory / f"worker{rank}.log").open("w")
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(devices), HF_HUB_OFFLINE="1")
            worker = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            workers.append(worker)
            report["workers"].append({"rank": rank, "devices": devices, "pid": worker.pid,
                                      "task_ids": [t["id"] for t in group], "command": command})
        while any(w.poll() is None for w in workers):
            if any(w.poll() not in (None, 0) for w in workers):
                raise RuntimeError("parallel worker failed; see worker logs")
            time.sleep(0.2)
        if any(w.returncode != 0 for w in workers):
            raise RuntimeError("parallel worker failed; see worker logs")
        report["passed"] = True
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        for worker in workers:
            try:
                worker.wait(timeout=15)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
        for entry, worker in zip(report["workers"], workers):
            entry["exit_code"] = worker.returncode
        for log in logs:
            log.close()
        report["wall_seconds_including_load"] = time.perf_counter() - started
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
