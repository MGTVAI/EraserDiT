"""Resolve model service contracts at the pipeline assembly boundary."""

from __future__ import annotations

from config.service_contract import PipelineServiceContract


def resolve_service_contract(pipeline_name: str | None) -> PipelineServiceContract:
    """Resolve the contract owned by the named pipeline."""
    from pipelines.registry import PipelineRegistry

    pipeline_cls, _ = PipelineRegistry.resolve(pipeline_name)
    contract = getattr(pipeline_cls, "service_contract", None)
    if contract is None:
        raise ValueError(
            f"pipeline {pipeline_cls.__name__} does not declare a service contract"
        )
    return contract
