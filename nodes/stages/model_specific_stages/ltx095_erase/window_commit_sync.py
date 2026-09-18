"""Stage adapter for the videoerase window commit protocol."""

from __future__ import annotations

from config.server_args import ServerArgs
from nodes.schedule_batch import Req
from nodes.stages.base import PipelineStage
from videoerase.windowing.commit_sync import (
    synchronize_ltx095_window_commit,
)


class LTX095EraseWindowCommitSyncStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        return synchronize_ltx095_window_commit(batch, server_args)


__all__ = ("LTX095EraseWindowCommitSyncStage",)
