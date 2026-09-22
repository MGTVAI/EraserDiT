"""Base class for composed pipelines in the minimal EraserDiT runtime."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import torch

from config.server_args import ServerArgs, set_global_server_args
from loader.component_loaders.component_loader import PipelineComponentLoader
from nodes.executors.pipeline_executor import PipelineExecutor
from nodes.executors.sync_executor import SyncExecutor
from nodes.schedule_batch import OutputBatch, Req
from nodes.stages.base import PipelineStage
from utils.hf_diffusers_utils import (
    maybe_download_model,
    verify_model_config_and_directory,
)
from utils.logging_utils import init_logger
from utils.platform import current_platform

logger = init_logger(__name__)


class ComposedPipelineBase(ABC):
    """Base class for pipelines composed of multiple stages."""

    is_video_pipeline: bool = False
    _required_config_modules: list[str] = []
    _extra_config_module_map: dict[str, str] = {}
    # Adapters whose checkpoint lacks ``model_index.json`` declare the component
    # map explicitly (plan §4.1): ``{module_name: (library, architecture)}``.
    _declared_model_index: dict[str, Any] | None = None
    server_args: ServerArgs | None = None
    modules: dict[str, Any] = {}
    executor: PipelineExecutor | None = None
    pipeline_name: str

    def __init__(
        self,
        model_path: str,
        server_args: ServerArgs,
        required_config_modules: list[str] | None = None,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
        executor: PipelineExecutor | None = None,
    ) -> None:
        self.server_args = server_args
        set_global_server_args(server_args)
        self.model_path = model_path
        self._stages: list[PipelineStage] = []
        self._stage_name_mapping: dict[str, PipelineStage] = {}
        self.executor = executor or self.build_executor(server_args)
        if required_config_modules is not None:
            self._required_config_modules = required_config_modules
        self.memory_usages: dict[str, float] = {}
        self.modules = self.load_modules(server_args, loaded_modules)
        self.__post_init__()

    def is_lora_effective(self) -> bool:
        return False

    def is_lora_set(self) -> bool:
        return False

    def build_executor(self, server_args: ServerArgs) -> PipelineExecutor:
        return SyncExecutor(server_args=server_args)

    def __post_init__(self) -> None:
        assert self.server_args is not None
        self.initialize_pipeline(self.server_args)
        self.create_pipeline_stages(self.server_args)

    def get_module(self, module_name: str, default_value: Any = None) -> Any:
        return self.modules.get(module_name, default_value)

    def add_module(self, module_name: str, module: Any) -> None:
        self.modules[module_name] = module

    def _load_config(self) -> dict[str, Any]:
        self.model_path = maybe_download_model(
            self.model_path, force_diffusers_model=True
        )
        if self._declared_model_index is not None:
            return {name: list(entry) for name, entry in self._declared_model_index.items()}
        return verify_model_config_and_directory(self.model_path)

    @property
    def required_config_modules(self) -> list[str]:
        return self._required_config_modules

    @property
    def stages(self) -> list[PipelineStage]:
        return self._stages

    @abstractmethod
    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        raise NotImplementedError

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        del server_args
        return None

    def _resolve_component_path(
        self, server_args: ServerArgs, module_name: str, load_module_name: str
    ) -> str:
        override_path = server_args.component_paths.get(module_name)
        if override_path is not None:
            return maybe_download_model(override_path)
        return os.path.join(self.model_path, load_module_name)

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        model_index = self._load_config()
        model_index.pop('_class_name', None)
        model_index.pop('_diffusers_version', None)

        required_modules = list(self.required_config_modules)
        loaded_components: dict[str, Any] = {}
        for module_name in required_modules:
            if loaded_modules is not None and module_name in loaded_modules:
                loaded_components[module_name] = loaded_modules[module_name]
                continue

            load_module_name = self._extra_config_module_map.get(module_name, module_name)
            if module_name not in model_index:
                raise ValueError(
                    f"Required module '{module_name}' missing from model_index.json"
                )

            transformers_or_diffusers, _architecture = model_index[module_name]
            component_model_path = self._resolve_component_path(
                server_args, module_name, load_module_name
            )
            module, memory_usage = PipelineComponentLoader.load_component(
                component_name=load_module_name,
                component_model_path=component_model_path,
                transformers_or_diffusers=transformers_or_diffusers,
                server_args=server_args,
            )
            self.memory_usages[load_module_name] = memory_usage
            loaded_components[module_name] = module

        logger.info(
            'Loaded required components: %s, avail mem: %.2f GB',
            required_modules,
            current_platform.get_available_gpu_memory(),
        )
        return loaded_components

    @staticmethod
    def _infer_stage_name(stage: PipelineStage) -> str:
        return stage.__class__.__name__

    def add_stage(
        self, stage: PipelineStage, stage_name: str | None = None
    ) -> 'ComposedPipelineBase':
        if stage_name is None:
            stage_name = self._infer_stage_name(stage)
        if stage_name in self._stage_name_mapping:
            raise ValueError(f'Duplicate stage name detected: {stage_name}')
        self._stages.append(stage)
        self._stage_name_mapping[stage_name] = stage
        return self

    def add_stages(
        self, stages: list[PipelineStage | tuple[PipelineStage, str]]
    ) -> 'ComposedPipelineBase':
        for item in stages:
            if isinstance(item, tuple):
                stage, name = item
                self.add_stage(stage, name)
            else:
                self.add_stage(item)
        return self

    def get_stage(self, stage_name: str) -> PipelineStage | None:
        return self._stage_name_mapping.get(stage_name)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch | Req:
        if not batch.is_warmup and not batch.suppress_logs:
            logger.info(
                'Running pipeline stages: %s', list(self._stage_name_mapping.keys())
            )
        assert self.executor is not None
        return self.executor.execute_with_profiling(self.stages, batch, server_args)
