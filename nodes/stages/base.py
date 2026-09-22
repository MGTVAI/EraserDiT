"""Base stage types for the minimal EraserDiT runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto

import torch

from config.server_args import ServerArgs, get_global_server_args
from nodes.schedule_batch import Req
from nodes.stages.validators import VerificationResult
from parallel.stage_policy import StageExecutionPolicy
from utils.logging_utils import init_logger
from utils.perf_logger import StageProfiler
from utils.platform import current_platform

logger = init_logger(__name__)


class StageParallelismType(Enum):
    REPLICATED = auto()
    MAIN_RANK_ONLY = auto()
    CFG_PARALLEL = auto()


class StageVerificationError(Exception):
    """Raised when a stage input/output contract is violated."""


class PipelineStage(ABC):
    """A discrete stage in the minimal EraserDiT runtime pipeline."""

    def __init__(self) -> None:
        self.server_args = get_global_server_args()
        self._enable_logging = True

    def log_info(self, msg: str, *args) -> None:
        if self._enable_logging:
            logger.info("[%s] " + msg, self.__class__.__name__, *args)

    def log_warning(self, msg: str, *args) -> None:
        logger.warning("[%s] " + msg, self.__class__.__name__, *args)

    def log_error(self, msg: str, *args) -> None:
        logger.error("[%s] " + msg, self.__class__.__name__, *args)

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        return VerificationResult()

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        return VerificationResult()

    def maybe_free_model_hooks(self) -> None:
        return None

    def load_model(self) -> None:
        return None

    def offload_model(self) -> None:
        return None

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.REPLICATED

    @property
    def execution_policy(self) -> StageExecutionPolicy:
        return StageExecutionPolicy.replicated()

    @property
    def device(self) -> torch.device:
        return torch.device(current_platform.device_type)

    def set_logging(self, enable: bool) -> None:
        self._enable_logging = enable

    def _run_verification(
        self,
        verification_result: VerificationResult,
        stage_name: str,
        verification_type: str,
    ) -> None:
        if verification_result.is_valid():
            return
        failed_fields = verification_result.get_failed_fields()
        if not failed_fields:
            return
        detailed_summary = verification_result.get_failure_summary()
        failed_fields_str = ", ".join(failed_fields)
        raise StageVerificationError(
            f"{verification_type.capitalize()} verification failed for {stage_name}: "
            f"Failed fields: {failed_fields_str}. Details: {detailed_summary}"
        )

    def __call__(self, batch: Req, server_args: ServerArgs) -> Req:
        stage_name = self.__class__.__name__
        input_result = self.verify_input(batch, server_args)
        self._run_verification(input_result, stage_name, "input")
        runtime_progress_enabled = bool(
            batch.extra.get("runtime_progress_enabled", False)
        )
        with StageProfiler(
            stage_name,
            logger=logger,
            metrics=batch.metrics,
            log_stage_start_end=not batch.is_warmup and not runtime_progress_enabled,
            perf_dump_path_provided=batch.perf_dump_path is not None,
        ):
            result = self.forward(batch, server_args)
        output_result = self.verify_output(result, server_args)
        self._run_verification(output_result, stage_name, "output")
        return result

    @abstractmethod
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        raise NotImplementedError
