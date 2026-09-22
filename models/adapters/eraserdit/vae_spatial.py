"""VAE spatial activation partitioning with layerwise convolution halo exchange.

Reference convolution shapes preserve cuDNN algorithm selection. Other ranks'
interior rows are zeros; only the local rows and immediate halo can influence
retained output rows. This duplicates convolution and normalization FLOPs while preserving full
spatial context. Memory and speed improvements must be measured independently.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier
import time

import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.normalization import RMSNorm


class SpatialExchange:
    def __init__(self, degree):
        self.degree = degree
        self.slots = [None] * degree
        self.barrier = Barrier(degree, timeout=120)
        self.calls = [0] * degree

    def abort(self):
        self.barrier.abort()

    def convolution(self, rank, value, original, radius):
        torch.cuda.current_stream(value.device).synchronize()
        self.slots[rank] = value
        self.barrier.wait()
        try:
            heights = [t.shape[-2] for t in self.slots]
            start, height = sum(heights[:rank]), heights[rank]
            before = self.slots[rank - 1][..., -radius:, :].to(value.device) if rank and radius else None
            after = self.slots[rank + 1][..., :radius, :].to(value.device) if rank + 1 < self.degree and radius else None
            tensors = ([before] if before is not None else []) + [value] + ([after] if after is not None else [])
            local = torch.cat(tensors, dim=-2)
            prefix = start - (before.shape[-2] if before is not None else 0)
            suffix = sum(heights) - prefix - local.shape[-2]
            padded = torch.nn.functional.pad(local, (0, 0, prefix, suffix))
            # Consumers must finish copying before a peer reuses its slot.
            torch.cuda.current_stream(value.device).synchronize()
            self.barrier.wait()
            result = original(padded)[..., start:start + height, :].contiguous()
            self.calls[rank] += 1
            return result
        except BaseException:
            self.abort()
            raise


def spatial_vae(vae, inputs, plan, batch, *, operation, temb=None):
    from models.vaes.eraserdit_vae import LTXVideoCausalConv3d, LTXVideoDownsampler3d
    encode = operation == "encode"
    component = vae.encoder if encode else vae.decoder
    ratio = vae.spatial_compression_ratio if encode else 1
    height = inputs.shape[-2]
    if height % ratio:
        raise ValueError("spatial VAE input height must align to its compression ratio")
    units = height // ratio
    degree = min(plan["vae"], units)
    if degree < 1:
        raise ValueError("spatial VAE requires nonempty height")
    if any(getattr(vae.config, "decoder_inject_noise", ())):
        raise ValueError("spatial VAE requires decoder_inject_noise disabled")
    if any(isinstance(m, torch.nn.GroupNorm) for m in component.modules()):
        raise ValueError("spatial VAE does not support global spatial GroupNorm")
    for m in component.modules():
        if isinstance(m, LTXVideoCausalConv3d) and (
            m.conv.stride[1] != 1 or m.conv.dilation[1] != 1 or m.kernel_size[1] % 2 != 1
        ):
            raise ValueError("spatial VAE requires odd kernels and unit convolution height stride/dilation")
    if degree == 1:
        output = component(inputs) if encode else component(inputs, temb)
        batch.extra[f"vae_parallel_{operation}"] = {
            "algorithm": "spatial_halo_reference", "requested_degree": plan["vae"],
            "effective_degree": 1, "fallback_reason": "insufficient_spatial_units"}
        return DiagonalGaussianDistribution(output) if encode else output
    devices = plan["devices"][:degree]
    exchange = SpatialExchange(degree)
    models, executor = [], None
    started = time.perf_counter()
    torch.cuda.current_stream(inputs.device).synchronize()
    try:
        for index, device in enumerate(devices):
            models.append(component if index == 0 else deepcopy(component).to(device).eval())
            torch.cuda.synchronize(device)
        setup_seconds = time.perf_counter() - started
        executor = ThreadPoolExecutor(max_workers=degree, thread_name_prefix="vae-spatial")
        def forward(rank):
            saved = []
            downsamplers = []
            try:
                with torch.cuda.device(devices[rank]), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for module in models[rank].modules():
                        if isinstance(module, LTXVideoDownsampler3d):
                            module._parallel_spatial_layout = (units, units * rank // degree,
                                                               units * (rank + 1) // degree)
                            downsamplers.append(module)
                        if isinstance(module, RMSNorm):
                            previous = module.__dict__.get("forward")
                            original = module.forward
                            saved.append((module, previous))
                            def norm(value, original=original):
                                # Restore NCTHW contiguous storage before the
                                # channel-last reduction, including global strides.
                                start_unit = units * rank // degree
                                end_unit = units * (rank + 1) // degree
                                factor = value.shape[-3] // (end_unit - start_unit)
                                padded = torch.nn.functional.pad(value.movedim(-1, 1).contiguous(),
                                    (0, 0, start_unit * factor, (units - end_unit) * factor))
                                result = original(padded.movedim(1, -1))
                                return result[..., start_unit * factor:end_unit * factor, :, :].movedim(-1, 1).contiguous().movedim(1, -1)
                            module.forward = norm
                        if not isinstance(module, LTXVideoCausalConv3d):
                            continue
                        previous = module.__dict__.get("forward")
                        original, radius = module.forward, module.kernel_size[1] // 2
                        saved.append((module, previous))
                        def conv(value, original=original, radius=radius):
                            return exchange.convolution(rank, value, original, radius)
                        module.forward = conv
                    start = units * rank // degree * ratio
                    end = units * (rank + 1) // degree * ratio
                    value = inputs[..., start:end, :].to(devices[rank]).contiguous()
                    output = models[rank](value) if encode else models[rank](value, temb.to(devices[rank]) if temb is not None else None)
                    torch.cuda.current_stream(devices[rank]).synchronize()
                    return output
            except BaseException:
                exchange.abort()
                raise
            finally:
                for module in downsamplers:
                    del module._parallel_spatial_layout
                for module, previous in saved:
                    if previous is None:
                        del module.forward
                    else:
                        module.forward = previous
        futures = [executor.submit(forward, rank) for rank in range(degree)]
        output = torch.cat([future.result().to(inputs.device) for future in futures], dim=-2)
        batch.extra[f"vae_parallel_{operation}"] = {
            "algorithm": "spatial_halo_reference", "requested_degree": plan["vae"],
            "effective_degree": degree, "devices": [str(d) for d in devices],
            "convolutions_per_rank": exchange.calls[:], "setup_seconds": setup_seconds,
            "full_spatial_context": True, "preserves_convolution_shape": True,
            "preserves_normalization_layout": True, "seconds": time.perf_counter() - started,
        }
        return DiagonalGaussianDistribution(output) if encode else output
    finally:
        exchange.abort()
        if executor is not None:
            executor.shutdown(wait=True)
        models.clear()
        exchange.slots.clear()
