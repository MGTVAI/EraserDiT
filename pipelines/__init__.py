"""Pipeline exports, loaded lazily so runtime helpers can be imported independently."""

__all__ = ["EraserDiTErasePipeline"]


def __getattr__(name: str):
    if name == "EraserDiTErasePipeline":
        from pipelines.eraserdit_erase_pipeline import EraserDiTErasePipeline

        globals()[name] = EraserDiTErasePipeline
        return EraserDiTErasePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
