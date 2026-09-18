"""Minimal profiler facade used by the pipeline executor."""

from __future__ import annotations


class SGLDiffusionProfiler:
    """A tiny singleton-compatible stub used during Phase 0."""

    _instance = None

    def __init__(
        self,
        request_id: str | None = None,
        rank: int = 0,
        full_profile: bool = False,
        num_steps: int | None = None,
        num_inference_steps: int | None = None,
    ) -> None:
        self.request_id = request_id
        self.rank = rank
        self.full_profile = full_profile
        self.num_steps = num_steps
        self.num_inference_steps = num_inference_steps
        self.has_stopped = False
        SGLDiffusionProfiler._instance = self

    @classmethod
    def get_instance(cls):
        return cls._instance

    def step_stage(self) -> None:
        return None

    def step_denoising_step(self) -> None:
        return None

    def stop(self, dump_rank: int | None = None) -> None:
        self.has_stopped = True
        if SGLDiffusionProfiler._instance is self:
            SGLDiffusionProfiler._instance = None
