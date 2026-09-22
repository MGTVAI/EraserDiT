"""Experimental EraserDiT policies; disabled by default.

Active-mode thresholds are sweep starting points, not quality-approved presets.
See docs/performance.md#cache for measured limits.
"""
from dataclasses import dataclass
from typing import ClassVar

from config.cache_dit import CacheDitParams
from config.teacache import TeaCacheParams, TeaCacheCoefficientSelection
from config.transformer_cache import validate_transformer_cache_request

MODEL_IDENTITY = 'jieeliu/EraserDiT'
TEACACHE_POLICY = 'eraserdit_modulated_input_relative_l1_experimental'
CACHE_DEFAULTS = {
    'transformer_cache_mode': 'off',
    'transformer_cache_force_compute': False,
    'cache_residual_predictor': 'none',
    'cache_text_projections': None,
    'teacache_threshold': 0.3,
    'max_teacache_consecutive_skip': 1,
    'teacache_warmup_steps': 4,
    'cache_dit_front_blocks': 1,
    'cache_dit_back_blocks': 0,
    'cache_dit_warmup_steps': 4,
    'cache_dit_residual_diff_threshold': 0.3,
    'cache_dit_max_consecutive_cached_steps': 1,
    'cache_end_guard_steps': 1,
}


@dataclass(frozen=True)
class EraserDiTTeaCacheParams(TeaCacheParams):
    coefficient_policy: str = TEACACHE_POLICY
    supported_coefficient_policies: ClassVar[tuple[str, ...]] = (TEACACHE_POLICY,)


def select_eraserdit_coefficients(sequence_length):
    # Identity polynomial: accumulate raw first-block modulated-input relative L1.
    # This is an explicit experimental policy, NOT a fitted model calibration.
    return TeaCacheCoefficientSelection(
        policy=TEACACHE_POLICY, model_identity=MODEL_IDENTITY,
        requested_global_sequence_length=sequence_length,
        selected_fit_length=sequence_length, fit_length_distance=0,
        complexity=1, calibration_size=0, coefficients=(1.0, 0.0),
    )


def resolve_eraserdit_cache_params(source, *, enable_torch_compile=False, num_blocks=28):
    def get(name):
        return source.get(name, CACHE_DEFAULTS[name]) if isinstance(source, dict) else getattr(source, name, CACHE_DEFAULTS[name])
    mode = validate_transformer_cache_request(
        mode=get('transformer_cache_mode'), enable_torch_compile=enable_torch_compile,
    )
    if get('cache_text_projections') is not None and type(get('cache_text_projections')) is not bool:
        raise TypeError('cache_text_projections must be a bool or None (auto)')
    if get('cache_text_projections') and enable_torch_compile:
        raise ValueError('EraserDiT text cache cannot be combined with torch.compile')
    if get('cache_residual_predictor') not in ('none', 'linear'):
        raise ValueError('cache_residual_predictor must be none or linear')
    force = get('transformer_cache_force_compute')
    if type(force) is not bool:
        raise TypeError('transformer_cache_force_compute must be a bool')
    guard = get('cache_end_guard_steps')
    if type(guard) is not int or guard < 1:
        raise ValueError('cache_end_guard_steps must be an integer >= 1')
    tea = EraserDiTTeaCacheParams(
        enabled=mode.value == 'teacache', threshold=get('teacache_threshold'),
        max_consecutive_skip=get('max_teacache_consecutive_skip'),
        min_skip_step=get('teacache_warmup_steps'), end_guard_steps=guard,
        calibrate=force,
    )
    dbc = CacheDitParams(
        enabled=mode.value == 'cache_dit', front_blocks=get('cache_dit_front_blocks'),
        back_blocks=get('cache_dit_back_blocks'), warmup_steps=get('cache_dit_warmup_steps'),
        residual_diff_threshold=get('cache_dit_residual_diff_threshold'),
        max_consecutive_cached_steps=get('cache_dit_max_consecutive_cached_steps'),
        end_guard_steps=guard,
    )
    if mode.value == 'cache_dit':
        dbc.validate_block_count(num_blocks)
    return mode, tea, dbc, force
