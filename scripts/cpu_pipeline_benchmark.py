"""Use the existing S3 benchmark with a metadata-only CLI observer."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path.cwd()))
if __package__:
    from . import benchmark_l40s
    from .cpu_pipeline_config import benchmark_environment
else:
    import benchmark_l40s
    from cpu_pipeline_config import benchmark_environment
original_popen = subprocess.Popen
observer = Path(__file__).with_name("cpu_pipeline_audit.py")


def launched(command, *args, **kwargs):
    if isinstance(command, (list, tuple)) and "-m" in command:
        index = command.index("-m")
        if len(command) > index + 1 and command[index + 1] == "entrypoints.cli.erase_eraserdit":
            command = [*command[:index], str(observer), *command[index + 2:]]
    return original_popen(command, *args, **kwargs)


def main(*, extra_env=None):
    subprocess.Popen = launched
    try:
        environment = {**benchmark_environment(), **(extra_env or {})}
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--run-dir', type=Path)
        args, _ = parser.parse_known_args()
        if args.run_dir is not None:
            environment.setdefault('ERASERDIT_CPU_PIPELINE_AUDIT',
                os.environ.get('ERASERDIT_CPU_PIPELINE_AUDIT')
                or str(args.run_dir.resolve()/'cpu_pipeline_audit.json'))
        benchmark_l40s.main(extra_env=environment)
    finally:
        subprocess.Popen = original_popen


if __name__ == "__main__":
    main()
