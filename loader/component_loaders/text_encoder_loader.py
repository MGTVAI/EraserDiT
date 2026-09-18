"""Minimal LTX095 text encoder and tokenizer loader."""

from __future__ import annotations

from typing import Any

from config.server_args import ServerArgs
from loader.component_loaders.component_loader import ComponentLoader

class TextEncoderLoader(ComponentLoader):
    """Loader for tokenizer and text encoder components.

    Supports LTX095 ``T5Tokenizer`` + ``T5EncoderModel`` components.
    """

    component_names = ["tokenizer", "text_encoder"]
    expected_library = "transformers"

    def should_offload(self, server_args: ServerArgs) -> bool:
        return server_args.resolve_resource_policy().text_encoder_cpu_offload

    def load_component(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str,
        transformers_or_diffusers: str,
        dtype,
    ) -> tuple[Any, dict[str, Any] | None]:
        del transformers_or_diffusers
        common_kwargs = {
            "trust_remote_code": server_args.trust_remote_code,
            "revision": server_args.revision,
            "local_files_only": True,
        }
        if component_name == "tokenizer":
            # Prefer the fast implementation and retain the standard tokenizer fallback.
            try:
                from transformers import T5TokenizerFast

                tokenizer = T5TokenizerFast.from_pretrained(
                    component_model_path, **common_kwargs
                )
                return tokenizer, None
            except Exception:
                from transformers import T5Tokenizer

                tokenizer = T5Tokenizer.from_pretrained(
                    component_model_path, **common_kwargs
                )
                return tokenizer, None

        if component_name != "text_encoder":
            raise ValueError(f"Unsupported text component: {component_name}")

        from transformers import T5EncoderModel

        text_encoder, loading_info = T5EncoderModel.from_pretrained(
            component_model_path,
            torch_dtype=dtype,
            output_loading_info=True,
            **common_kwargs,
        )
        return text_encoder, loading_info
