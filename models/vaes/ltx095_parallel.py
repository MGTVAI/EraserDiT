"""LTX-specific spatial planning and merge adapter for common VAE parallelism."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any

import torch

from parallel.vae_parallel import (
    VAEOperation,
    VAETaskPlan,
    VAETaskSpec,
    validate_vae_task_plan,
)

LTX095_ENCODE_OVERLAP_LATENT_UNITS = 4
LTX095_DECODE_OVERLAP_LATENT_UNITS = 4
LTX095_VAE_PARALLEL_MIN_SAMPLE_SPATIAL_SIZE = 640


@dataclass(frozen=True)
class VAEExecutionDecision:
    operation: VAEOperation
    resolved_degree: int
    effective_degree: int
    sample_height: int
    sample_width: int
    fallback_reason: str | None


class LTX095IncrementalMerger:
    """Merge row-major tiles while retaining only the active overlap frontier."""

    def __init__(
        self,
        adapter: "LTX095VAEParallelAdapter",
        plan: VAETaskPlan,
    ) -> None:
        validate_vae_task_plan(plan)
        self.adapter = adapter
        self.plan = plan
        self._by_id = {task.task_id: task for task in plan.tasks}
        positions = {
            task.task_id: (
                task.valid_output_slices[-2][0],
                task.valid_output_slices[-1][0],
            )
            for task in plan.tasks
        }
        self._row_starts = sorted({position[0] for position in positions.values()})
        self._column_starts = sorted(
            {position[1] for position in positions.values()}
        )
        self._position_by_id = positions
        self._task_by_position = {
            position: self._by_id[task_id] for task_id, position in positions.items()
        }
        self._pending: dict[int, torch.Tensor] = {}
        self._raw_frontier: dict[tuple[int, int], torch.Tensor] = {}
        self._next_task_id = 0
        self._merged: torch.Tensor | None = None
        self.peak_retained_tile_count = 0

    @property
    def complete(self) -> bool:
        return self._next_task_id == len(self.plan.tasks)

    def __call__(
        self,
        current: torch.Tensor | None,
        outputs: tuple[tuple[VAETaskSpec, torch.Tensor], ...],
    ) -> torch.Tensor:
        if type(outputs) is not tuple:
            raise TypeError("outputs must be a tuple")
        for task, output in outputs:
            if task.task_id not in self._by_id or self._by_id[task.task_id] != task:
                raise ValueError("merge output task is not in the plan")
            if task.task_id < self._next_task_id or task.task_id in self._pending:
                raise ValueError("merge outputs contain duplicate task_id")
            self._pending[task.task_id] = self.adapter._validated_output(task, output)
        self._drain()
        if self._merged is None:
            if current is None:
                raise RuntimeError("incremental merge must receive task 0 first")
            return current
        return self._merged

    def _drain(self) -> None:
        while self._next_task_id in self._pending:
            task = self._by_id[self._next_task_id]
            raw = self._pending.pop(self._next_task_id)
            position = self._position_by_id[task.task_id]
            row_index = self._row_starts.index(position[0])
            column_index = self._column_starts.index(position[1])
            self._raw_frontier = {
                key: value
                for key, value in self._raw_frontier.items()
                if self._row_starts.index(key[0]) >= row_index - 1
            }
            if self._merged is None:
                self._merged = torch.zeros(
                    self.plan.global_output_shape,
                    dtype=raw.dtype,
                    device=raw.device,
                )
            tile = raw.clone()
            current_origin = self.adapter._task_output_origin(self.plan, task)
            if row_index > 0:
                above_position = (
                    self._row_starts[row_index - 1],
                    position[1],
                )
                above = self._task_by_position[above_position]
                above_origin = self.adapter._task_output_origin(self.plan, above)
                extent = (
                    above_origin[-2]
                    + above.padded_output_shape[-2]
                    - current_origin[-2]
                )
                tile = _blend_vertical(
                    self._raw_frontier[above_position],
                    tile,
                    extent,
                )
                self._raw_frontier.pop(above_position)
            if column_index > 0:
                left_position = (
                    position[0],
                    self._column_starts[column_index - 1],
                )
                left = self._task_by_position[left_position]
                left_origin = self.adapter._task_output_origin(self.plan, left)
                extent = (
                    left_origin[-1]
                    + left.padded_output_shape[-1]
                    - current_origin[-1]
                )
                tile = _blend_horizontal(
                    self._raw_frontier[left_position],
                    tile,
                    extent,
                )
                if row_index + 1 == len(self._row_starts):
                    self._raw_frontier.pop(left_position)
            self.adapter._copy_valid_output(self.plan, task, tile, self._merged)
            self._raw_frontier[position] = raw
            self.peak_retained_tile_count = max(
                self.peak_retained_tile_count,
                len(self._raw_frontier) + len(self._pending),
            )
            self._next_task_id += 1


def choose_spatial_grid(
    worker_count: int,
    *,
    height: int,
    width: int,
) -> tuple[int, int]:
    if type(worker_count) is not int or worker_count <= 0:
        raise ValueError("worker_count must be a positive plain int")
    if worker_count & (worker_count - 1):
        raise ValueError("worker_count must be a power-of-two")
    if type(height) is not int or type(width) is not int or min(height, width) <= 0:
        raise ValueError("height and width must be positive plain ints")
    smaller = 2 ** (int(math.log2(worker_count)) // 2)
    larger = worker_count // smaller
    return (smaller, larger) if width > height else (larger, smaller)


def assign_tasks_by_cost(
    task_costs: tuple[tuple[int, int], ...],
    group_ranks: tuple[int, ...],
) -> dict[int, int]:
    if not group_ranks:
        raise ValueError("group_ranks must not be empty")
    if any(type(rank) is not int or rank < 0 for rank in group_ranks):
        raise ValueError("group_ranks must contain non-negative plain ints")
    if len(set(group_ranks)) != len(group_ranks):
        raise ValueError("group_ranks must be unique")
    task_ids = tuple(task_id for task_id, _ in task_costs)
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task IDs must be unique")
    for task_id, cost in task_costs:
        if type(task_id) is not int or task_id < 0:
            raise ValueError("task IDs must be non-negative plain ints")
        if type(cost) is not int or cost <= 0:
            raise ValueError("task costs must be positive plain ints")
    loads = {rank: 0 for rank in group_ranks}
    assignment: dict[int, int] = {}
    for task_id, cost in sorted(task_costs, key=lambda item: (-item[1], item[0])):
        rank = min(group_ranks, key=lambda item: (loads[item], item))
        assignment[task_id] = rank
        loads[rank] += cost
    return assignment


class LTX095VAEParallelAdapter:
    def __init__(
        self,
        vae: Any,
        *,
        encode_sample_overlap: int,
        decode_latent_overlap: int,
    ) -> None:
        self.vae = vae
        self.spatial_ratio = _positive_plain_int(
            "spatial_compression_ratio",
            getattr(vae, "spatial_compression_ratio", None),
        )
        self.temporal_ratio = _positive_plain_int(
            "temporal_compression_ratio",
            getattr(vae, "temporal_compression_ratio", None),
        )
        self.encode_sample_overlap = _positive_plain_int(
            "encode_sample_overlap",
            encode_sample_overlap,
        )
        if self.encode_sample_overlap % self.spatial_ratio != 0:
            raise ValueError(
                "encode_sample_overlap must align to spatial_compression_ratio"
            )
        self.decode_latent_overlap = _positive_plain_int(
            "decode_latent_overlap",
            decode_latent_overlap,
        )

    def resolve_execution_decision(
        self,
        *,
        operation: VAEOperation,
        input_shape: tuple[int, ...],
        resolved_degree: int,
    ) -> VAEExecutionDecision:
        if not isinstance(operation, VAEOperation):
            raise TypeError("operation must be a VAEOperation")
        if (
            type(input_shape) is not tuple
            or len(input_shape) != 5
            or any(type(value) is not int or value <= 0 for value in input_shape)
        ):
            raise ValueError("input_shape must contain positive B/C/T/H/W dimensions")
        resolved_degree = _positive_plain_int("resolved_degree", resolved_degree)
        input_height, input_width = input_shape[-2:]
        if operation is VAEOperation.DECODE:
            sample_height = input_height * self.spatial_ratio
            sample_width = input_width * self.spatial_ratio
        else:
            sample_height = input_height
            sample_width = input_width
        if resolved_degree == 1:
            effective_degree = 1
            fallback_reason = "resolved_degree_one"
        elif (
            sample_height <= LTX095_VAE_PARALLEL_MIN_SAMPLE_SPATIAL_SIZE
            or sample_width <= LTX095_VAE_PARALLEL_MIN_SAMPLE_SPATIAL_SIZE
        ):
            effective_degree = 1
            fallback_reason = "spatial_threshold"
        else:
            effective_degree = resolved_degree
            fallback_reason = None
        return VAEExecutionDecision(
            operation=operation,
            resolved_degree=resolved_degree,
            effective_degree=effective_degree,
            sample_height=sample_height,
            sample_width=sample_width,
            fallback_reason=fallback_reason,
        )

    def build_encode_plan(
        self,
        sample: torch.Tensor,
        *,
        group_ranks: tuple[int, ...],
        moments_channels: int,
        grid_shape: tuple[int, int] | None = None,
    ) -> VAETaskPlan:
        _validate_video_tensor("sample", sample)
        moments_channels = _positive_plain_int(
            "moments_channels",
            moments_channels,
        )
        batch, channels, frames, height, width = tuple(sample.shape)
        if height % self.spatial_ratio or width % self.spatial_ratio:
            raise ValueError(
                "sample spatial shape must align to spatial_compression_ratio"
            )
        latent_height = height // self.spatial_ratio
        latent_width = width // self.spatial_ratio
        latent_frames = math.ceil(frames / self.temporal_ratio)
        overlap = self.encode_sample_overlap // self.spatial_ratio
        grid = self._resolve_grid(
            group_ranks,
            height=latent_height,
            width=latent_width,
            grid_shape=grid_shape,
        )
        if grid_shape is None:
            grid = _choose_overlap_aware_spatial_grid(
                len(group_ranks),
                height=latent_height,
                width=latent_width,
                overlap=overlap,
            )
        return self._build_plan(
            operation=VAEOperation.ENCODE,
            group_ranks=group_ranks,
            grid_shape=grid,
            global_input_shape=(batch, channels, frames, height, width),
            global_output_shape=(
                batch,
                moments_channels,
                latent_frames,
                latent_height,
                latent_width,
            ),
            overlap=overlap,
            input_spatial_scale=self.spatial_ratio,
            output_spatial_scale=1,
        )

    def build_decode_plan(
        self,
        latents: torch.Tensor,
        *,
        group_ranks: tuple[int, ...],
        output_channels: int = 3,
        grid_shape: tuple[int, int] | None = None,
    ) -> VAETaskPlan:
        _validate_video_tensor("latents", latents)
        output_channels = _positive_plain_int("output_channels", output_channels)
        batch, channels, frames, height, width = tuple(latents.shape)
        sample_frames = 1 + (frames - 1) * self.temporal_ratio
        sample_height = height * self.spatial_ratio
        sample_width = width * self.spatial_ratio
        grid = self._resolve_grid(
            group_ranks,
            height=height,
            width=width,
            grid_shape=grid_shape,
        )
        return self._build_plan(
            operation=VAEOperation.DECODE,
            group_ranks=group_ranks,
            grid_shape=grid,
            global_input_shape=(batch, channels, frames, height, width),
            global_output_shape=(
                batch,
                output_channels,
                sample_frames,
                sample_height,
                sample_width,
            ),
            overlap=self.decode_latent_overlap,
            input_spatial_scale=1,
            output_spatial_scale=self.spatial_ratio,
        )

    @staticmethod
    def materialize_input(
        full_input: torch.Tensor,
        task: VAETaskSpec,
    ) -> torch.Tensor:
        if not isinstance(full_input, torch.Tensor):
            raise TypeError("full_input must be a torch.Tensor")
        if not isinstance(task, VAETaskSpec):
            raise TypeError("task must be a VAETaskSpec")
        slices = tuple(slice(start, end) for start, end in task.input_slices)
        return full_input[slices].contiguous()

    def execute_local_encode(
        self,
        local_input: torch.Tensor,
        task: VAETaskSpec,
    ) -> torch.Tensor:
        del task
        output = self.vae.encoder(local_input)
        if not isinstance(output, torch.Tensor):
            raise TypeError("LTX encoder must return a torch.Tensor")
        return output.contiguous()

    def execute_local_decode(
        self,
        local_input: torch.Tensor,
        task: VAETaskSpec,
        *,
        temb: torch.Tensor | None,
    ) -> torch.Tensor:
        del task
        output = self.vae.decoder(local_input, temb)
        if hasattr(output, "sample"):
            output = output.sample
        if not isinstance(output, torch.Tensor):
            raise TypeError("LTX decoder must return a torch.Tensor")
        return output.contiguous()

    def merge_encode(
        self,
        plan: VAETaskPlan,
        outputs: tuple[tuple[VAETaskSpec, torch.Tensor], ...],
    ) -> torch.Tensor:
        if plan.operation is not VAEOperation.ENCODE:
            raise ValueError("merge_encode requires an encode plan")
        return self._merge(plan, outputs)

    def merge_decode(
        self,
        plan: VAETaskPlan,
        outputs: tuple[tuple[VAETaskSpec, torch.Tensor], ...],
    ) -> torch.Tensor:
        if plan.operation is not VAEOperation.DECODE:
            raise ValueError("merge_decode requires a decode plan")
        return self._merge(plan, outputs)

    def create_incremental_merger(
        self,
        plan: VAETaskPlan,
    ) -> LTX095IncrementalMerger:
        return LTX095IncrementalMerger(self, plan)

    def _resolve_grid(
        self,
        group_ranks: tuple[int, ...],
        *,
        height: int,
        width: int,
        grid_shape: tuple[int, int] | None,
    ) -> tuple[int, int]:
        if type(group_ranks) is not tuple or not group_ranks:
            raise ValueError("group_ranks must be a non-empty tuple")
        if grid_shape is None:
            grid_shape = choose_spatial_grid(
                len(group_ranks),
                height=height,
                width=width,
            )
        if (
            type(grid_shape) is not tuple
            or len(grid_shape) != 2
            or any(type(value) is not int or value <= 0 for value in grid_shape)
        ):
            raise ValueError("grid_shape must contain two positive plain ints")
        if grid_shape[0] > height or grid_shape[1] > width:
            raise ValueError("grid_shape cannot create empty spatial tiles")
        if grid_shape[0] * grid_shape[1] < len(group_ranks):
            raise ValueError("grid_shape must provide at least one task per rank")
        return grid_shape

    def _build_plan(
        self,
        *,
        operation: VAEOperation,
        group_ranks: tuple[int, ...],
        grid_shape: tuple[int, int],
        global_input_shape: tuple[int, ...],
        global_output_shape: tuple[int, ...],
        overlap: int,
        input_spatial_scale: int,
        output_spatial_scale: int,
    ) -> VAETaskPlan:
        latent_height = global_input_shape[-2] // input_spatial_scale
        latent_width = global_input_shape[-1] // input_spatial_scale
        height_bounds = _overlap_balanced_partition_bounds(
            latent_height,
            grid_shape[0],
            overlap,
        )
        width_bounds = _overlap_balanced_partition_bounds(
            latent_width,
            grid_shape[1],
            overlap,
        )
        task_fields: list[dict[str, Any]] = []
        task_id = 0
        for row in range(grid_shape[0]):
            height_start, height_valid_end = height_bounds[row : row + 2]
            height_crop_start = max(0, height_start - overlap)
            height_crop_end = min(latent_height, height_valid_end + overlap)
            for column in range(grid_shape[1]):
                width_start, width_valid_end = width_bounds[column : column + 2]
                width_crop_start = max(0, width_start - overlap)
                width_crop_end = min(latent_width, width_valid_end + overlap)
                input_slices = (
                    (0, global_input_shape[0]),
                    (0, global_input_shape[1]),
                    (0, global_input_shape[2]),
                    (
                        height_crop_start * input_spatial_scale,
                        height_crop_end * input_spatial_scale,
                    ),
                    (
                        width_crop_start * input_spatial_scale,
                        width_crop_end * input_spatial_scale,
                    ),
                )
                output_slices = (
                    (0, global_output_shape[0]),
                    (0, global_output_shape[1]),
                    (0, global_output_shape[2]),
                    (
                        height_start * output_spatial_scale,
                        height_valid_end * output_spatial_scale,
                    ),
                    (
                        width_start * output_spatial_scale,
                        width_valid_end * output_spatial_scale,
                    ),
                )
                padded_input_shape = tuple(end - start for start, end in input_slices)
                padded_output_shape = (
                    global_output_shape[0],
                    global_output_shape[1],
                    global_output_shape[2],
                    (height_crop_end - height_crop_start) * output_spatial_scale,
                    (width_crop_end - width_crop_start) * output_spatial_scale,
                )
                task_fields.append(
                    {
                        "task_id": task_id,
                        "input_slices": input_slices,
                        "valid_output_slices": output_slices,
                        "padded_input_shape": padded_input_shape,
                        "padded_output_shape": padded_output_shape,
                        "estimated_cost": _shape_numel(padded_input_shape),
                    }
                )
                task_id += 1
        assignment = assign_tasks_by_cost(
            tuple((item["task_id"], item["estimated_cost"]) for item in task_fields),
            group_ranks,
        )
        tasks = tuple(
            VAETaskSpec(
                assigned_rank=assignment[item["task_id"]],
                **item,
            )
            for item in task_fields
        )
        counts = tuple(
            sum(task.assigned_rank == rank for task in tasks) for rank in group_ranks
        )
        plan = VAETaskPlan(
            operation=operation,
            owner_rank=group_ranks[0],
            global_input_shape=global_input_shape,
            global_output_shape=global_output_shape,
            tasks=tasks,
            rounds=max(counts),
        )
        return validate_vae_task_plan(plan)

    def _merge(
        self,
        plan: VAETaskPlan,
        outputs: tuple[tuple[VAETaskSpec, torch.Tensor], ...],
    ) -> torch.Tensor:
        validate_vae_task_plan(plan)
        if type(outputs) is not tuple:
            raise TypeError("outputs must be a tuple")
        by_id: dict[int, torch.Tensor] = {}
        for task, output in outputs:
            if task not in plan.tasks:
                raise ValueError("merge output task is not in the plan")
            if task.task_id in by_id:
                raise ValueError("merge outputs contain duplicate task_id")
            by_id[task.task_id] = self._validated_output(task, output)
        if set(by_id) != {task.task_id for task in plan.tasks}:
            raise ValueError("merge outputs must contain every task exactly once")

        first = by_id[0]
        merged = torch.zeros(
            plan.global_output_shape,
            dtype=first.dtype,
            device=first.device,
        )
        task_by_position = {
            (
                task.valid_output_slices[-2][0],
                task.valid_output_slices[-1][0],
            ): task
            for task in plan.tasks
        }
        row_starts = sorted({position[0] for position in task_by_position})
        column_starts = sorted({position[1] for position in task_by_position})
        for row_index, height_start in enumerate(row_starts):
            for column_index, width_start in enumerate(column_starts):
                task = task_by_position[(height_start, width_start)]
                tile = by_id[task.task_id].clone()
                if row_index > 0:
                    above = task_by_position[(row_starts[row_index - 1], width_start)]
                    above_tile = by_id[above.task_id]
                    above_origin = self._task_output_origin(plan, above)
                    current_origin = self._task_output_origin(plan, task)
                    extent = (
                        above_origin[-2]
                        + above.padded_output_shape[-2]
                        - current_origin[-2]
                    )
                    tile = _blend_vertical(above_tile, tile, extent)
                if column_index > 0:
                    left = task_by_position[
                        (height_start, column_starts[column_index - 1])
                    ]
                    left_tile = by_id[left.task_id]
                    left_origin = self._task_output_origin(plan, left)
                    current_origin = self._task_output_origin(plan, task)
                    extent = (
                        left_origin[-1]
                        + left.padded_output_shape[-1]
                        - current_origin[-1]
                    )
                    tile = _blend_horizontal(left_tile, tile, extent)
                self._copy_valid_output(plan, task, tile, merged)
        return merged.contiguous()

    @staticmethod
    def _validated_output(
        task: VAETaskSpec,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(output, torch.Tensor):
            raise TypeError("merge output must be a torch.Tensor")
        expected = task.padded_output_shape
        if output.ndim != len(expected) or any(
            actual < required
            for actual, required in zip(output.shape, expected, strict=True)
        ):
            raise ValueError("merge output is smaller than padded_output_shape")
        return output[tuple(slice(0, size) for size in expected)].contiguous()

    def _copy_valid_output(
        self,
        plan: VAETaskPlan,
        task: VAETaskSpec,
        tile: torch.Tensor,
        merged: torch.Tensor,
    ) -> None:
        valid_shape = tuple(
            end - start for start, end in task.valid_output_slices
        )
        output_origin = self._task_output_origin(plan, task)
        source_offsets = tuple(
            start - origin
            for (start, _), origin in zip(
                task.valid_output_slices,
                output_origin,
                strict=True,
            )
        )
        source = tuple(
            slice(offset, offset + length)
            for offset, length in zip(source_offsets, valid_shape, strict=True)
        )
        destination = tuple(
            slice(start, end) for start, end in task.valid_output_slices
        )
        merged[destination] = tile[source]

    def _task_output_origin(
        self,
        plan: VAETaskPlan,
        task: VAETaskSpec,
    ) -> tuple[int, ...]:
        if plan.operation is VAEOperation.ENCODE:
            spatial_scale = self.spatial_ratio
            height_origin = task.input_slices[-2][0] // spatial_scale
            width_origin = task.input_slices[-1][0] // spatial_scale
        elif plan.operation is VAEOperation.DECODE:
            spatial_scale = self.spatial_ratio
            height_origin = task.input_slices[-2][0] * spatial_scale
            width_origin = task.input_slices[-1][0] * spatial_scale
        else:
            raise ValueError(f"unsupported LTX VAE operation: {plan.operation}")
        return (0, 0, 0, height_origin, width_origin)


def _blend_vertical(
    above: torch.Tensor,
    current: torch.Tensor,
    extent: int,
) -> torch.Tensor:
    extent = min(max(extent, 0), above.shape[-2], current.shape[-2])
    for index in range(extent):
        weight = index / extent
        current[..., index, :] = torch.lerp(
            above[..., -extent + index, :],
            current[..., index, :],
            weight,
        )
    return current


def _blend_horizontal(
    left: torch.Tensor,
    current: torch.Tensor,
    extent: int,
) -> torch.Tensor:
    extent = min(max(extent, 0), left.shape[-1], current.shape[-1])
    for index in range(extent):
        weight = index / extent
        current[..., index] = torch.lerp(
            left[..., -extent + index],
            current[..., index],
            weight,
        )
    return current


def _partition_bounds(total: int, parts: int) -> tuple[int, ...]:
    return tuple((index * total) // parts for index in range(parts + 1))


def _overlap_balanced_partition_bounds(
    total: int,
    parts: int,
    overlap: int,
) -> tuple[int, ...]:
    if parts <= 2:
        return _partition_bounds(total, parts)
    padded_extent = (total + 2 * overlap * (parts - 1)) / parts
    interior_extent = padded_extent - 2 * overlap
    if interior_extent <= 0:
        return _partition_bounds(total, parts)
    first_extent = padded_extent - overlap
    bounds = [0]
    for index in range(1, parts):
        ideal = first_extent + (index - 1) * interior_extent
        lower = bounds[-1] + 1
        upper = total - (parts - index)
        bounds.append(min(max(int(math.floor(ideal + 0.5)), lower), upper))
    bounds.append(total)
    return tuple(bounds)


def _choose_overlap_aware_spatial_grid(
    worker_count: int,
    *,
    height: int,
    width: int,
    overlap: int,
) -> tuple[int, int]:
    base_grid = choose_spatial_grid(
        worker_count,
        height=height,
        width=width,
    )
    candidates: list[tuple[tuple[int, int, int], tuple[int, int]]] = []
    for rows in range(1, worker_count + 1):
        if worker_count % rows:
            continue
        columns = worker_count // rows
        if rows > height or columns > width:
            continue
        height_bounds = _overlap_balanced_partition_bounds(height, rows, overlap)
        width_bounds = _overlap_balanced_partition_bounds(width, columns, overlap)
        padded_heights = tuple(
            min(height, end + overlap) - max(0, start - overlap)
            for start, end in zip(
                height_bounds[:-1],
                height_bounds[1:],
                strict=True,
            )
        )
        padded_widths = tuple(
            min(width, end + overlap) - max(0, start - overlap)
            for start, end in zip(
                width_bounds[:-1],
                width_bounds[1:],
                strict=True,
            )
        )
        areas = tuple(
            padded_height * padded_width
            for padded_height in padded_heights
            for padded_width in padded_widths
        )
        candidates.append(
            (
                (
                    max(areas),
                    sum(areas),
                    0 if (rows, columns) == base_grid else 1,
                ),
                (rows, columns),
            )
        )
    if not candidates:
        raise ValueError("worker_count cannot create non-empty spatial tiles")
    return min(candidates)[1]


def _shape_numel(shape: tuple[int, ...]) -> int:
    return reduce(mul, shape, 1)


def _positive_plain_int(name: str, value: Any) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a plain int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_video_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 5 or any(dimension <= 0 for dimension in value.shape):
        raise ValueError(f"{name} must have non-empty B/C/T/H/W dimensions")


__all__ = (
    "LTX095_DECODE_OVERLAP_LATENT_UNITS",
    "LTX095_ENCODE_OVERLAP_LATENT_UNITS",
    "LTX095VAEParallelAdapter",
    "LTX095_VAE_PARALLEL_MIN_SAMPLE_SPATIAL_SIZE",
    "VAEExecutionDecision",
    "assign_tasks_by_cost",
    "choose_spatial_grid",
)
