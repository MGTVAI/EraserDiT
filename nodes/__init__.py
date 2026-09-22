"""Execution exports; control helpers can be imported without loading models."""

__all__ = ["ComposedPipelineBase", "Req"]


def __getattr__(name: str):
    if name == "ComposedPipelineBase":
        from nodes.composed_pipeline_base import ComposedPipelineBase

        return ComposedPipelineBase
    if name == "Req":
        from nodes.schedule_batch import Req

        return Req
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
