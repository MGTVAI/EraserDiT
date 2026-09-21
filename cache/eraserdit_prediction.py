"""EraserDiT residual forecasting, inspired by SGLang's TaylorSeer option.

Only real block evaluations train the first-order predictor. Residual arithmetic
uses float32 so subtracting and restoring bf16 activations does not add another
rounding error. The block interface always retains the original activation dtype.
"""
from dataclasses import dataclass

import torch

from cache.cache_dit import CacheDitController
from cache.teacache import TeaCacheController


@dataclass
class _Forecast:
    step: int | None = None
    layout: str | None = None
    interval: int = 0
    velocity: torch.Tensor | None = None
    predicted_steps: int = 0


class _PredictionMixin:
    def __init__(self, *args, residual_predictor="none", **kwargs):
        super().__init__(*args, **kwargs)
        self.residual_predictor = residual_predictor
        self._forecasts = {branch: _Forecast() for branch in self._states}

    @staticmethod
    def _compute_residual(input_hidden_states, output_hidden_states):
        return (output_hidden_states.float() - input_hidden_states.float()).detach()

    def _learn(self, branch, step, previous, residual, reason, layout):
        state = self._forecasts[branch]
        state.velocity = None
        if (self.residual_predictor == "linear" and previous is not None
                and state.step is not None and step > state.step
                and state.layout == layout and previous.shape == residual.shape
                and reason not in ("layout_changed", "non_contiguous_branch_step")):
            state.interval = step - state.step
            state.velocity = (residual - previous) / state.interval
        state.step = step
        state.layout = layout

    def _predict(self, branch, step, output, dtype):
        state = self._forecasts[branch]
        if state.velocity is not None:
            # Bound extrapolation to one observed interval, including when a
            # caller allows a long run of cache hits. Never train on predictions.
            horizon = min(step - state.step, state.interval)
            output = output + state.velocity * horizon
            state.predicted_steps += 1
        return output.to(dtype)

    def stats(self):
        report = super().stats()
        report["residual_predictor"] = self.residual_predictor
        report["residual_dtype"] = "torch.float32"
        report["predicted_steps"] = sum(s.predicted_steps for s in self._forecasts.values())
        return report

    def _close(self, *, abort_reason):
        try:
            return super()._close(abort_reason=abort_reason)
        finally:
            for state in self._forecasts.values():
                state.velocity = None


class EraserDiTTeaCacheController(_PredictionMixin, TeaCacheController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._probe_trace = {branch: [] for branch in self._states}

    def check(self, **kwargs):
        decision = super().check(**kwargs)
        self._probe_trace[kwargs['branch']].append({
            'step': kwargs['step'], 'reason': decision.reason,
            'estimated_distance': decision.estimated_distance,
            'accumulated_distance': decision.accumulated_distance,
        })
        return decision

    def stats(self):
        report = super().stats()
        report['probe'] = 'first_block_modulated_input'
        report['probe_trace'] = {branch.value: list(trace) for branch, trace in self._probe_trace.items()}
        return report

    def record_compute(self, *, branch, step, input_hidden_states, output_hidden_states, **kwargs):
        self._require_open()
        state = self._states[branch]
        if state.pending is None or state.pending.should_skip or state.pending_step != step:
            raise RuntimeError("TeaCache compute result has no pending compute decision")
        previous, reason = state.previous_residual, state.pending.reason
        layout = state.pending_layout_signature
        super().record_compute(branch=branch, step=step, input_hidden_states=input_hidden_states,
                               output_hidden_states=output_hidden_states, **kwargs)
        residual = state.previous_residual
        self._learn(branch, step, previous, residual, reason, layout)

    def apply_cached_update(self, *, branch, step, input_hidden_states, **kwargs):
        output = super().apply_cached_update(branch=branch, step=step,
                                             input_hidden_states=input_hidden_states, **kwargs)
        return self._predict(branch, step, output, input_hidden_states.dtype)


class EraserDiTCacheDitController(_PredictionMixin, CacheDitController):
    def record_middle_compute(self, *, branch, step, front_output_hidden_states, middle_output_hidden_states):
        self._require_pending(branch, step, should_reuse=False)
        state = self._states[branch]
        previous, reason = state.cached_middle_residual, state.pending.reason
        layout = state.pending_layout_signature
        super().record_middle_compute(branch=branch, step=step,
                                      front_output_hidden_states=front_output_hidden_states,
                                      middle_output_hidden_states=middle_output_hidden_states)
        residual = state.cached_middle_residual
        self._learn(branch, step, previous, residual, reason, layout)

    def apply_cached_middle(self, *, branch, step, front_output_hidden_states):
        output = super().apply_cached_middle(branch=branch, step=step,
                                            front_output_hidden_states=front_output_hidden_states)
        return self._predict(branch, step, output, front_output_hidden_states.dtype)
