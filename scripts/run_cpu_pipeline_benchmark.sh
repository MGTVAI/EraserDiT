#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' 'Usage: bash scripts/run_cpu_pipeline_benchmark.sh [--gpu N] [--runs N] [--run-dir DIR] [--reference-sha256 HASH] [--prepare-only]'
  printf '%s\n' 'Default: GPU 0, two independent processes, full fixed S3 sample. Prepare-only freezes code/config without using a GPU.'
}

gpu=0
runs=2
run_dir=
reference=
prepare_only=0
while (($#)); do
  case "$1" in
    --gpu|--runs|--run-dir|--reference-sha256)
      if (($# < 2)); then usage >&2; exit 2; fi
      case "$1" in
        --gpu) gpu=$2;;
        --runs) runs=$2;;
        --run-dir) run_dir=$2;;
        --reference-sha256) reference=$2;;
      esac
      shift 2;;
    --prepare-only) prepare_only=1; shift;;
    --help|-h) usage; exit 0;;
    *) usage >&2; exit 2;;
  esac
done
if [[ ! $gpu =~ ^[0-9]+$ || ! $runs =~ ^[1-9][0-9]*$ ]]; then usage >&2; exit 2; fi
if [[ -n $reference && ! $reference =~ ^[[:xdigit:]]{64}$ ]]; then usage >&2; exit 2; fi
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
source scripts/activate_cpu_pipeline.sh
unset ERASERDIT_NSYS_CAPTURE ERASERDIT_RUNTIME_SIDECAR MGERASE_DIAGNOSTIC_TIMING ERASERDIT_PREP_DIAGNOSTIC_JSON ERASERDIT_DEBUG_DUMP
export ERASERDIT_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
if [[ -z $run_dir ]]; then run_dir="results/cpu_pipeline_$(date +%Y%m%d_%H%M%S)"; fi
if [[ $run_dir != /* ]]; then run_dir="$repo_dir/$run_dir"; fi
if [[ -e $run_dir ]]; then printf 'Run directory already exists: %s\n' "$run_dir" >&2; exit 2; fi
mkdir -p "$run_dir/reproduction"
printf 'RESULT_DIR=%s\n' "$run_dir"
cp scripts/activate_cpu_pipeline.sh scripts/cpu_pipeline_audit.py scripts/cpu_pipeline_benchmark.py scripts/run_cpu_pipeline_benchmark.sh scripts/summarize_cpu_pipeline.py "$run_dir/reproduction/"
git rev-parse HEAD > "$run_dir/commit.txt"
git diff > "$run_dir/source_diff.patch"
git status --short > "$run_dir/git_status.txt"
uv run --no-project python -m entrypoints.cli.benchmark_l40s --profile s3 --devices "$gpu" --cache off --repeats 1 --prepare-only --run-dir "$run_dir/prepared" > "$run_dir/prepare.log" 2>&1
if ((prepare_only)); then printf 'Prepared only; no inference launched.\n'; exit 0; fi
for ((run=1; run<=runs; run++)); do
  export ERASERDIT_CPU_PIPELINE_AUDIT="$run_dir/run_$run-runtime.json"
  printf 'START run_%s\n' "$run"
  uv run --no-project python "$run_dir/reproduction/cpu_pipeline_benchmark.py" --profile s3 --devices "$gpu" --cache off --repeats 1 --source-root "$run_dir/prepared/source" --run-dir "$run_dir/run_$run" > "$run_dir/run_$run-driver.log" 2>&1
  printf 'success\n' > "$run_dir/run_${run}_COMPLETE"
done
uv run --no-project python scripts/summarize_cpu_pipeline.py "$run_dir" --runs "$runs" --reference-sha256 "$reference"
printf 'success\n' > "$run_dir/ALL_COMPLETE"
