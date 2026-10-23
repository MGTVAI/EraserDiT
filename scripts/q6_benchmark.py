"""Run tracked Q6 presets with S4 through the current benchmark adapter."""
from pathlib import Path
import hashlib
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_l40s, cpu_pipeline_benchmark
from scripts.q6_presets import preset_environment


def main():
    base = benchmark_l40s.PROFILES['q6']
    for name in ('q6a', 'q6b', 'q6c'):
        options = list(base[1])
        index = options.index('--operator-fusion-ops')+1
        options[index] += ',cross_q_rmsnorm_native'
        benchmark_l40s.PROFILES[name] = (1, options)
    if '--help' in sys.argv or '-h' in sys.argv:
        return benchmark_l40s.main()
    profile = sys.argv[sys.argv.index('--profile')+1] if '--profile' in sys.argv else None
    env = preset_environment(profile)
    if profile == 'q6c':
        run_dir = Path(sys.argv[sys.argv.index('--run-dir')+1]).resolve()
        env['MGERASE_Q6_QKV_CALIBRATION'] = str(run_dir/'source/config/calibration/q6_qkv_113000356_seed42.json')
        path = Path(env['MGERASE_Q6_FFN_LIBRARY'])
        env['MGERASE_Q6_FFN_LIBRARY_SHA256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    cpu_pipeline_benchmark.main(extra_env=env)


if __name__ == '__main__':
    main()
