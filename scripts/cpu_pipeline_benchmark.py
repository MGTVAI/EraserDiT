"""Use the existing S3 benchmark with a metadata-only CLI observer."""
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path.cwd()))
from entrypoints.cli import benchmark_l40s
from cpu_pipeline_config import benchmark_environment
original_popen = subprocess.Popen
observer = Path(__file__).with_name("cpu_pipeline_audit.py")


def launched(command, *args, **kwargs):
    if isinstance(command, (list, tuple)) and "-m" in command:
        index = command.index("-m")
        if len(command) > index + 1 and command[index + 1] == "entrypoints.cli.erase_eraserdit":
            command = [*command[:index], str(observer), *command[index + 2:]]
    return original_popen(command, *args, **kwargs)


def main():
    subprocess.Popen = launched
    try:
        benchmark_l40s.main(extra_env=benchmark_environment())
    finally:
        subprocess.Popen = original_popen


if __name__ == "__main__":
    main()
