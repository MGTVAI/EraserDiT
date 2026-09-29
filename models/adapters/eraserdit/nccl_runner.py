"""Rank-local DiT computation; scheduler and RNG remain in the owner pipeline."""
from contextlib import nullcontext

import torch
import torch.distributed as dist

from distributed.dit_groups import gather_variable
from models.adapters.eraserdit.nccl_sequence import DistributedSequenceRank


class DiTRankRunner:
    def __init__(self, model, groups, config):
        self.model, self.groups, self.config = model, groups, config
        self.sequence = DistributedSequenceRank(groups, ring_mode=config.ring_attention_mode) if groups.topology.sp > 1 else None
        self.static = None
        self.rotary = None
        self.forwards = 0
        for block in model.transformer_blocks:
            block.attn1.processor.sequence_parallel = self.sequence

    def reset(self):
        self.static = self.rotary = None

    def predict(self, packet):
        if packet['static'] is not None:
            self.static = packet['static']
            self.rotary = None
        hidden, timestep = packet['hidden'], packet['timestep']
        if self.static is None:
            raise RuntimeError('first step requires window static conditions')
        topology = self.groups.topology
        cfg_rank = self.groups.coordinates[3]
        branches = ('positive', 'negative') if topology.cfg == 1 else (('positive',) if cfg_rank == 0 else ('negative',))
        outputs = {}
        with torch.no_grad(), torch.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=hidden.is_cuda):
            for branch in branches:
                values = self.static[branch]
                if self.rotary is None:
                    self.rotary = self.model.rope(hidden, values['num_frames'], values['height'], values['width'],
                                                  values.get('rope_interpolation_scale'), values.get('video_coords'))
                scope = self.sequence.linear_scope(self.model, self.config.sp_linear_mode) if self.sequence else nullcontext()
                with scope:
                    output = self.model(**values, hidden_states=hidden, timestep=timestep,
                                        image_rotary_emb=self.rotary, sequence_parallel=self.sequence)[0]
                if self.sequence:
                    group, ranks, _ = self.groups.get('sp')
                    lengths = tuple(self.sequence._length(i) for i in range(len(ranks)))
                    output = gather_variable(output, group, len(ranks), dim=1, lengths=lengths)
                    from models.dits.eraserdit_transformer import unpack_latents
                    output = unpack_latents(output, values['num_frames'], values['height'], values['width'],
                                            self.model.config.patch_size, self.model.config.patch_size_t)
                outputs[branch] = output.float()
                self.forwards += 1
        if topology.cfg == 2:
            group, ranks, _ = self.groups.get('cfg')
            local = outputs[branches[0]].contiguous()
            pair = [torch.empty_like(local) for _ in ranks]
            dist.all_gather(pair, local, group=group)
            return pair[1], pair[0]
        return outputs['negative'], outputs['positive']

    def report(self):
        device = next(self.model.parameters()).device
        weight_bytes = sum((p.to_local() if hasattr(p, 'to_local') else p).numel() * p.element_size()
                           for p in self.model.parameters())
        return dict(rank=self.groups.rank, coordinates=self.groups.coordinates,
                    tp_linear_mode=self.config.tp_linear_mode, sp_linear_mode=self.config.sp_linear_mode,
                    additional_nccl_groups=len(self.groups.owned),
                    successful_forwards=self.forwards, local_parameter_bytes=weight_bytes,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0,
                    ring_partial_dtype=self.sequence.partial_attention_dtype if self.sequence else None,
                    actual_self_attention=self.sequence.effective_attention if self.sequence else 'torch_sdpa',
                    collectives=self.sequence.calls if self.sequence else {})
