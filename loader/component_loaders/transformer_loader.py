"""Minimal video erase transformer loader."""

from __future__ import annotations

import os
from typing import Any

from config.server_args import ServerArgs
from loader.component_loaders.component_loader import ComponentLoader
from models.registry import ModelRegistry
from utils.hf_diffusers_utils import load_json_dict


class TransformerLoader(ComponentLoader):
    """Loader for the local video erase transformer."""

    component_names = ["transformer"]
    expected_library = "diffusers"

    def should_offload(self, server_args: ServerArgs) -> bool:
        return server_args.dit_cpu_offload or server_args.dit_layerwise_offload

    def load_component(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str,
        transformers_or_diffusers: str,
        dtype,
    ) -> tuple[Any, dict[str, Any] | None]:
        del transformers_or_diffusers
        config_path = os.path.join(component_model_path, "config.json")
        config = load_json_dict(config_path)
        architecture = server_args.component_architectures.get(
            component_name
        ) or config.get("_class_name", "LTXVideoTransformer3DModel")
        model_cls, _ = ModelRegistry.resolve_model_cls(architecture)
        from loader.meta_load import load_safetensors_model
        model, loading_info = load_safetensors_model(
            lambda: model_cls.from_config(config), component_model_path, dtype=dtype,
        )
        addition_config = load_json_dict(
            os.path.join(component_model_path, "addition_config.json")
        )
        setattr(model, "addition_config", addition_config)
        return model, loading_info
