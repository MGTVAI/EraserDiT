"""Cumulative single-GPU presets. Apply S4 explicitly in the adapter only."""
from pathlib import Path
import os
from scripts.cpu_pipeline_config import S4_ENV

COMMON = dict(MGERASE_Q6_SAGE_PERIPHERAL='0', MGERASE_Q6_TARGETED_FILL='0',
              MGERASE_Q6_QKV_CALIBRATION='', MGERASE_Q6_FFN_EPILOGUE='0',
              ERASERDIT_DETERMINISTIC='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')


def preset_environment(profile):
    if profile not in ('q6', 'q6a', 'q6b', 'q6c'):
        raise ValueError('select q6, q6a, q6b or q6c')
    env = {**S4_ENV, **COMMON}
    if profile != 'q6':
        env['MGERASE_Q6_SAGE_PERIPHERAL'] = '1'
    if profile in ('q6b', 'q6c'):
        env['MGERASE_Q6_TARGETED_FILL'] = '1'
    if profile == 'q6c':
        path = Path(__file__).resolve().parents[1]/'config/calibration/q6_qkv_113000356_seed42.json'
        env['MGERASE_Q6_QKV_CALIBRATION'] = str(path)
        env['MGERASE_Q6_FFN_EPILOGUE'] = '1'
        library = os.environ.get('MGERASE_Q6_FFN_LIBRARY', '')
        if not Path(library).is_file():
            raise ValueError('set MGERASE_Q6_FFN_LIBRARY to the built optional adapter')
        env['MGERASE_Q6_FFN_LIBRARY'] = str(Path(library).resolve())
    return env
