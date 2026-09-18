"""Synchronous executor for the minimal MGErase runtime."""

from __future__ import annotations

from config.server_args import ServerArgs
from nodes.executors.pipeline_executor import PipelineExecutor
from nodes.schedule_batch import OutputBatch, Req
from parallel.stage_policy import synchronize_stage_error
from utils.profiler import SGLDiffusionProfiler


class SyncExecutor(PipelineExecutor):
    """Execute all stages sequentially in-process."""

    def run_profile_all_stages(
        self,
        stages,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        runtime_progress_enabled = bool(
            batch.extra.get("runtime_progress_enabled", False)
        )
        parallel_context = getattr(server_args, "parallel_context", None)
        if parallel_context is None:
            global_rank = 0
            world_size = 1
            groups = {}
        else:
            global_rank = int(parallel_context.global_rank)
            world_size = int(parallel_context.plan.world_size)
            groups = parallel_context.groups
        group_names = set(groups)
        member_groups = {name for name, group in groups.items() if group.is_member}
        for stage in stages:
            stage.set_logging(not runtime_progress_enabled)
            stage_error = None
            executed = False
            try:
                policy = stage.execution_policy
                policy.validate_context(
                    group_names=group_names,
                    world_size=world_size,
                )
                if policy.should_execute(
                    global_rank=global_rank,
                    group_names=member_groups,
                ):
                    batch = stage(batch, server_args)
                    executed = True
            except Exception as error:
                stage_error = error
            finally:
                stage.set_logging(True)

            synchronize_stage_error(stage_error, parallel_context)
            if executed:
                profiler = SGLDiffusionProfiler.get_instance()
                if profiler is not None:
                    profiler.step_stage()
        return batch

    def execute(
        self,
        stages,
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch | Req:
        return self.run_profile_all_stages(stages, batch, server_args)
