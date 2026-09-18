"""LTX095 cache-mode dispatcher and common window lifecycle."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, TYPE_CHECKING

from config.transformer_cache import (
    TransformerCacheMode,
    resolve_transformer_cache_mode,
)
from cache.cache_dit import (
    CacheDitController,
    build_ltx095_cache_dit_controller,
)
from cache.teacache import (
    TeaCacheController,
    build_ltx095_teacache_controller,
)

if TYPE_CHECKING:
    from distributed.group_coordinator import GroupCoordinator

LTX095TransformerCacheController = TeaCacheController | CacheDitController


def build_ltx095_transformer_cache_controller(
    *,
    batch: Any,
    total_steps: int,
    sp_degree: int = 1,
    sp_rank: int = 0,
    cfg_degree: int = 1,
    cfg_rank: int = 0,
    coordinator: GroupCoordinator | None = None,
    sp_group_identity: str | None = None,
    cfg_group_identity: str | None = None,
) -> LTX095TransformerCacheController | None:
    mode = resolve_transformer_cache_mode(
        getattr(batch, "transformer_cache_mode", "off")
    )
    if mode is TransformerCacheMode.OFF:
        return None
    common = {
        "batch": batch,
        "total_steps": total_steps,
        "sp_degree": sp_degree,
        "sp_rank": sp_rank,
        "cfg_degree": cfg_degree,
        "cfg_rank": cfg_rank,
        "coordinator": coordinator,
        "sp_group_identity": sp_group_identity,
        "cfg_group_identity": cfg_group_identity,
    }
    if mode is TransformerCacheMode.TEACACHE:
        return build_ltx095_teacache_controller(**common)
    return build_ltx095_cache_dit_controller(**common)


@contextmanager
def ltx095_transformer_cache_window_scope(
    batch: Any,
    controller: LTX095TransformerCacheController | None,
):
    if controller is None:
        yield None
        return
    try:
        yield controller
    except BaseException as error:
        batch.extra["transformer_cache"] = controller.abort_window(
            f"{type(error).__name__}: {error}"
        )
        raise
    else:
        batch.extra["transformer_cache"] = controller.finish_window()


__all__ = (
    "LTX095TransformerCacheController",
    "build_ltx095_transformer_cache_controller",
    "ltx095_transformer_cache_window_scope",
)
