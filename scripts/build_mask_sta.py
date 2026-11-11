"""Build the optional fixed-shape SM89 mask-STA adapter from repository sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda-home', type=Path, default=Path(os.environ.get('CUDA_HOME', '/usr/local/cuda')))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('the optional mask STA adapter currently supports Linux only')
    compiler = args.cuda_home/'bin/nvcc'
    if not compiler.is_file():
        parser.error(f'missing CUDA compiler: {compiler}')
    source = Path(__file__).resolve().parents[1]/'layers/attention/backends/csrc/mask_sta'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(compiler), '-std=c++17', '-O3', '--use_fast_math', '--expt-relaxed-constexpr',
        '-gencode=arch=compute_89,code=sm_89', '--shared', '-Xcompiler=-fPIC',
        '-Xlinker=-rpath', f'-Xlinker={args.cuda_home/"lib64"}',
        f'-I{source/"vendor"}', f'-I{source/"vendor/sparge"}', f'-I{source/"vendor/sparge/qattn"}',
        str(source/'core.cu'), str(source/'value.cu'), str(source/'quantize_value.cu'),
        '-o', str(args.output)]
    subprocess.run(command, check=True)
    inputs = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(source.rglob('*')) if path.is_file()}
    args.output.with_suffix('.build.json').write_text(json.dumps(dict(
        command=command, source_sha256=inputs,
        library_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest()), indent=2) + '\n')


if __name__ == '__main__':
    main()
