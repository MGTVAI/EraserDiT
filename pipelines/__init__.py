"""Pipeline exports, loaded lazily so runtime helpers can be imported independently."""

__all__ = ["LTX095ErasePipeline"]


def __getattr__(name: str):
    if name == "LTX095ErasePipeline":
        from pipelines.ltx_095_erase_pipeline import LTX095ErasePipeline

        globals()[name] = LTX095ErasePipeline
        return LTX095ErasePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
