"""Minimal video erase VAE loader."""

from __future__ import annotations

import os
from typing import Any

from config.server_args import ServerArgs
from loader.component_loaders.component_loader import ComponentLoader
from models.registry import ModelRegistry
from utils.hf_diffusers_utils import load_json_dict


class VAELoader(ComponentLoader):
    """Loader for the local video erase VAE."""

    component_names = ["vae"]
    expected_library = "diffusers"

    def should_offload(self, server_args: ServerArgs) -> bool:
        return server_args.resolve_resource_policy().vae_cpu_offload

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
        ) or config.get("_class_name", "AutoencoderKLLTXVideo")
        model_cls, _ = ModelRegistry.resolve_model_cls(architecture)
        vae, loading_info = model_cls.from_pretrained(
            component_model_path,
            torch_dtype=dtype,
            local_files_only=True,
            output_loading_info=True,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
        )
        return vae, loading_info
