"""Executor base classes for the minimal EraserDiT runtime."""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from config.server_args import ServerArgs
from nodes.schedule_batch import OutputBatch, Req
from utils.logging_utils import init_logger
from utils.perf_logger import StageProfiler
from utils.profiler import SGLDiffusionProfiler

if TYPE_CHECKING:
    from nodes.stages.base import PipelineStage

logger = init_logger(__name__)


class Timer(StageProfiler):
    """Backward-compatible alias for simple stage timing."""

    def __init__(self, name: str = "Stage") -> None:
        super().__init__(
            stage_name=name,
            logger=logger,
            metrics=None,
            log_stage_start_end=True,
        )


class PipelineExecutor(ABC):
    """Base executor for running a list of pipeline stages."""

    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args

    def execute_with_profiling(
        self,
        stages: list["PipelineStage"],
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch | Req:
        with self.profile_execution(batch):
            return self.execute(stages, batch, server_args)

    @abstractmethod
    def execute(
        self,
        stages: list["PipelineStage"],
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch | Req:
        raise NotImplementedError

    @contextlib.contextmanager
    def profile_execution(self, batch: Req):
        if not batch.profile or batch.is_warmup:
            yield
            return

        profiler = SGLDiffusionProfiler(
            request_id=batch.request_id,
            rank=0,
            full_profile=batch.profile_all_stages,
            num_steps=batch.num_profiled_timesteps,
            num_inference_steps=batch.num_inference_steps,
        )
        try:
            yield
        finally:
            profiler.stop(dump_rank=0)
