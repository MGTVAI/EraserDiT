"""Build the optional CUDA 12.6/cuBLASLt FFN adapter, outside the source tree."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda-home', type=Path, default=Path(os.environ.get('CUDA_HOME', '/usr/local/cuda')))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('the optional adapter currently supports Linux only')
    packages = Path(sysconfig.get_paths()['purelib'])/'nvidia'
    source = Path(__file__).resolve().parents[1]/'layers/quantization/csrc/ffn_epilogue.cpp'
    for path in (args.cuda_home/'include/cuda_runtime_api.h', packages/'cublas/include/cublasLt.h'):
        if not path.is_file():
            parser.error(f'missing build dependency: {path}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = ['g++', '-shared', '-fPIC', '-O3', '-std=c++17',
        '-I', str(packages/'cublas/include'), '-I', str(args.cuda_home/'include'), str(source),
        '-L', str(packages/'cublas/lib'), '-l:libcublasLt.so.12', '-o', str(args.output)]
    subprocess.run(command, check=True)
    args.output.with_suffix('.build.json').write_text(json.dumps(dict(command=command,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest()), indent=2)+'\n')


if __name__ == '__main__':
    main()
