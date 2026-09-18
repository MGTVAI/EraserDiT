"""EraserDiT erase – window commit sync stage.

The commit handshake is windowing-runtime behaviour, not model behaviour, so the
EraserDiT pipeline reuses the shared implementation unchanged.
"""

from __future__ import annotations

from config.server_args import ServerArgs
from nodes.schedule_batch import Req
from nodes.stages.base import PipelineStage
from videoerase.windowing.commit_sync import synchronize_ltx095_window_commit


class EraserDiTEraseWindowCommitSyncStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        return synchronize_ltx095_window_commit(batch, server_args)
