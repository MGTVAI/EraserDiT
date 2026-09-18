"""Compatibility shim for legacy custom-op imports."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn


class CustomOp(nn.Module):
    """Tiny stand-in for the old custom-op dispatch base."""

    op_registry: dict[str, type["CustomOp"]] = {}

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args, **kwargs) -> Any:
        if hasattr(self, "forward_native"):
            return self.forward_native(*args, **kwargs)
        raise NotImplementedError

    @classmethod
    def register(cls, name: str) -> Callable:
        def decorator(op_cls):
            cls.op_registry[name] = op_cls
            op_cls.name = name
            return op_cls

        return decorator

    @classmethod
    def enabled(cls) -> bool:
        return True
