"""Ulysses x Ring on orthogonal NCCL groups, with unequal-length support."""
import torch
import torch.distributed as dist

from models.adapters.eraserdit.mesh import SequenceRank


class DistributedSequenceRank(SequenceRank):
    def __init__(self, groups, *, ring_mode='reference'):
        self.groups = groups
        self.degree = groups.topology.sp
        self.rank = groups.coordinates[1] + groups.topology.ulysses * groups.coordinates[2]
        self.start = self.end = 0
        self.ring_mode = ring_mode
        self.calls = dict(ulysses=0, ring=0)
        self.partial_attention_dtype = None
        self.effective_attention = None

    def _length(self, sp_rank):
        return self.length * (sp_rank + 1) // self.degree - self.length * sp_rank // self.degree

    def _ulysses_input(self, value):
        group, ranks, urank = self.groups.get('ulysses')
        size = len(ranks)
        if size == 1:
            return value
        batch, local, heads, dim = value.shape
        if heads % size:
            raise ValueError('attention heads must be divisible by Ulysses degree')
        h = heads // size
        rrank = self.groups.coordinates[2]
        lengths = [self._length(rrank * size + u) for u in range(size)]
        inputs = [x.contiguous().flatten() for x in value.split(h, dim=2)]
        recv_counts = [batch * length * h * dim for length in lengths]
        output = value.new_empty(sum(recv_counts))
        dist.all_to_all_single(output, torch.cat(inputs), recv_counts,
                               [x.numel() for x in inputs], group=group)
        self.calls['ulysses'] += 1
        return torch.cat([x.view(batch, length, h, dim) for x, length in
                          zip(output.split(recv_counts), lengths)], dim=1)

    def _ulysses_output(self, value):
        group, ranks, urank = self.groups.get('ulysses')
        size = len(ranks)
        if size == 1:
            return value
        batch, _, heads, dim = value.shape
        rrank = self.groups.coordinates[2]
        lengths = [self._length(rrank * size + u) for u in range(size)]
        inputs = [x.contiguous().flatten() for x in value.split(lengths, dim=1)]
        count = batch * lengths[urank] * heads * dim
        output = value.new_empty(count * size)
        dist.all_to_all_single(output, torch.cat(inputs), [count] * size,
                               [x.numel() for x in inputs], group=group)
        self.calls['ulysses'] += 1
        return torch.cat([x.view(batch, lengths[urank], heads, dim)
                          for x in output.split(count)], dim=2)

    def _rotate(self, key, value, owner):
        group, ranks, rank = self.groups.get('ring')
        size = len(ranks)
        next_owner = (owner - 1) % size
        length = self.length * (next_owner + 1) // size - self.length * next_owner // size
        shape = (key.shape[0], length, key.shape[2], key.shape[3])
        recv_k, recv_v = key.new_empty(shape), value.new_empty(shape)
        key, value = key.contiguous(), value.contiguous()
        operations = [dist.P2POp(dist.isend, key, ranks[(rank + 1) % size], group),
                      dist.P2POp(dist.irecv, recv_k, ranks[(rank - 1) % size], group),
                      dist.P2POp(dist.isend, value, ranks[(rank + 1) % size], group),
                      dist.P2POp(dist.irecv, recv_v, ranks[(rank - 1) % size], group)]
        for work in dist.batch_isend_irecv(operations):
            work.wait()
        self.calls['ring'] += 1
        return recv_k, recv_v, next_owner

    def attention(self, query, key, value, impl, metadata):
        if impl.causal or impl.dropout or metadata.attn_mask is not None:
            raise ValueError('distributed DiT self-attention requires unmasked noncausal SDPA with dropout=0')
        if len(self.groups.get('ulysses')[1]) > 1:
            # Q/K/V have identical layouts in self-attention. Batch packing
            # keeps their reduction order unchanged while using one exchange.
            q, k, v = self._ulysses_input(torch.cat((query, key, value), dim=0)).chunk(3, dim=0)
        else:
            q, k, v = query, key, value
        _, ranks, owner = self.groups.get('ring')
        if len(ranks) == 1:
            self.effective_attention = 'torch_sdpa'
            output = impl.forward(q, k, v, metadata)
        elif self.ring_mode == 'reference':
            self.effective_attention = 'torch_sdpa_ring_gather'
            # Alignment path preserves full-K reduction order. Communication
            # is ring P2P, but this mode retains all KV shards; it does not claim
            # online Ring's reduced KV memory. Optimize only after SSIM gating.
            pieces = {owner: (k, v)}
            for _ in range(len(ranks) - 1):
                k, v, owner = self._rotate(k, v, owner)
                pieces[owner] = (k, v)
            full_k = torch.cat([pieces[i][0] for i in range(len(ranks))], dim=1)
            full_v = torch.cat([pieces[i][1] for i in range(len(ranks))], dim=1)
            output = impl.forward(q, full_k, full_v, metadata)
        elif self.ring_mode == 'streaming':
            self.effective_attention = 'triton_ring_fp32_fragments'
            from layers.attention.ring_accumulator import piece
            if not q.is_cuda or q.shape[-1] not in (16, 32, 64):
                raise ValueError('streaming Ring requires CUDA with head dimension 16, 32 or 64')
            state = carry_k = carry_v = None
            started = finished = False
            output = None
            self.partial_attention_dtype = 'float32_fragment_accumulator'
            # Every rank sees shards in the same descending global order.
            # Different ranks start on different hops; a second circuit lets
            # all ranks finish without buffering the whole sequence.
            for hop in range(2 * len(ranks) - 1):
                if owner == len(ranks) - 1:
                    started = True
                if started and not finished:
                    chunk_k = k if carry_k is None else torch.cat((k, carry_k), dim=1)
                    chunk_v = v if carry_v is None else torch.cat((v, carry_v), dim=1)
                    begin = self.length * owner // len(ranks)
                    skip = min((-begin) % 128, chunk_k.shape[1])
                    carry_k, carry_v = chunk_k[:, :skip].clone(), chunk_v[:, :skip].clone()
                    if chunk_k.shape[1] > skip:
                        result = piece(q, chunk_k[:, skip:], chunk_v[:, skip:], state,
                                       final=owner == 0, scale=impl.softmax_scale)
                        if owner == 0:
                            output = result
                        else:
                            state = result
                    finished = owner == 0
                    del chunk_k, chunk_v
                if hop + 1 < 2 * len(ranks) - 1:
                    k, v, owner = self._rotate(k, v, owner)
            if output is None:
                raise RuntimeError('streaming Ring did not finish the global sequence')
        else:
            self.effective_attention = 'torch_flash_ring_lse_merge'
            if not q.is_cuda or metadata.attn_mask is not None:
                raise ValueError('online Ring requires unmasked CUDA attention')
            # Experimental: native partial outputs round before the FP32 LSE
            # merge. This path failed the full-video 0.985 gate; reference is
            # deliberately the default. FP16 partials also failed and were not
            # adopted (see the distributed-parallel validation report).
            output_dtype = q.dtype
            self.partial_attention_dtype = str(q.dtype)
            accumulator = total_lse = None
            for step in range(len(ranks)):
                out, lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                    0., False, False, scale=impl.softmax_scale)
                if accumulator is None:
                    accumulator, total_lse = out.float(), lse
                else:
                    merged = torch.logaddexp(total_lse, lse)
                    accumulator.mul_(torch.exp(total_lse - merged).unsqueeze(-1))
                    accumulator.add_(out.float() * torch.exp(lse - merged).unsqueeze(-1))
                    total_lse = merged
                if step + 1 < len(ranks):
                    k, v, owner = self._rotate(k, v, owner)
            output = accumulator.to(output_dtype).transpose(1, 2).contiguous()
        return self._ulysses_output(output)
