"""Window-owned EraserDiT caches with independent CFG branch state."""
from dataclasses import replace
from uuid import uuid4

import torch

from cache.base import CacheBranch
from cache.cache_dit import CacheDitController
from cache.teacache import TeaCacheController
from config.eraserdit_cache import (
    MODEL_IDENTITY, resolve_eraserdit_cache_params, select_eraserdit_coefficients,
)


class EraserDiTCacheWindow:
    def __init__(self, batch, *, total_steps, num_blocks, enable_torch_compile=False):
        mode, tea, dbc, self.force_compute = resolve_eraserdit_cache_params(
            batch, enable_torch_compile=enable_torch_compile, num_blocks=num_blocks,
        )
        self.batch = batch
        self.mode = mode.value
        self.controller = None
        if self.mode == 'off':
            return
        extra = batch.extra
        common = dict(
            request_id=str(getattr(batch, 'request_id', None) or getattr(batch, 'rid', None)
                           or extra.get('request_id') or uuid4()),
            object_index=int(extra.get('object_index', 0)),
            window_index=int(extra.get('window_index', 0)), total_steps=total_steps,
            sp_degree=1, sp_rank=0, cfg_degree=1, cfg_rank=0, coordinator=None,
            sp_group_identity='eraserdit_serial', cfg_group_identity='eraserdit_serial',
            model_identity=MODEL_IDENTITY,
        )
        if self.mode == 'teacache':
            self.controller = TeaCacheController(
                tea, coefficient_selector=select_eraserdit_coefficients, **common,
            )
        else:
            if self.force_compute:
                dbc = replace(dbc, warmup_steps=max(total_steps, dbc.warmup_steps))
            self.controller = CacheDitController(dbc, num_transformer_blocks=num_blocks, **common)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.controller is None:
            report = {'mode': 'off'}
        elif exc_type is None:
            report = self.controller.finish_window()
        else:
            report = self.controller.abort_window(exc_type.__name__)
        report['peak_retained_tensor_bytes'] = getattr(self.controller, '_eraserdit_peak_retained_bytes', 0)
        report['force_compute'] = self.force_compute
        report['experimental'] = self.mode != 'off'
        if self.mode == 'teacache':
            report['coefficient_calibrated'] = False
        self.batch.extra['transformer_cache'] = report
        return False

    def kwargs(self, branch, step):
        if self.controller is None:
            return {}
        return {'cache_adapter': EraserDiTCacheBranch(self.controller, CacheBranch(branch), step)}


class EraserDiTCacheBranch:
    def __init__(self, controller, branch, step):
        self.controller, self.branch, self.step = controller, branch, step

    def run(self, hidden_states, temb, run_blocks, *, num_blocks, layout):
        try:
            return self._run(hidden_states, temb, run_blocks, num_blocks=num_blocks, layout=layout)
        finally:
            # Retained cache tensors only; allocator peaks include transient activations.
            storages = {}
            for state in self.controller._states.values():
                for value in vars(state).values():
                    if isinstance(value, torch.Tensor):
                        storage = value.untyped_storage()
                        storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
            self.controller._eraserdit_peak_retained_bytes = max(
                getattr(self.controller, '_eraserdit_peak_retained_bytes', 0), sum(storages.values()),
            )

    def _run(self, hidden_states, temb, run_blocks, *, num_blocks, layout):
        c = self.controller
        common = dict(
            branch=self.branch, step=self.step,
            global_sequence_length=hidden_states.shape[1],
            local_sequence_length=hidden_states.shape[1],
            valid_local_sequence_length=hidden_states.shape[1],
            layout_signature=repr((layout, tuple(hidden_states.shape), str(hidden_states.dtype), str(hidden_states.device))),
            dtype=str(hidden_states.dtype), device=str(hidden_states.device),
            hidden_width=hidden_states.shape[-1],
        )
        if isinstance(c, TeaCacheController):
            decision = c.check(modulated_input=temb, **common)
            if decision.should_skip:
                return c.apply_cached_update(
                    branch=self.branch, step=self.step, modulated_input=temb,
                    input_hidden_states=hidden_states,
                )
            output = run_blocks(hidden_states, 0, num_blocks)
            c.record_compute(
                branch=self.branch, step=self.step, modulated_input=temb,
                input_hidden_states=hidden_states, output_hidden_states=output,
            )
            return output
        front = run_blocks(hidden_states, 0, c.front_end)
        decision = c.check(input_hidden_states=hidden_states, front_output_hidden_states=front, **common)
        if decision.should_reuse_middle:
            middle = c.apply_cached_middle(branch=self.branch, step=self.step, front_output_hidden_states=front)
        else:
            middle = run_blocks(front, c.front_end, c.middle_end)
            c.record_middle_compute(
                branch=self.branch, step=self.step,
                front_output_hidden_states=front, middle_output_hidden_states=middle,
            )
        output = run_blocks(middle, c.back_start, num_blocks)
        c.complete_step(branch=self.branch, step=self.step)
        return output
