"""Persistent one-process-per-GPU DiT ranks, isolated from parent T5 FSDP.

CPU tensor IPC remains the default boundary. An opt-in CUDA IPC boundary copies
into receiver-owned GPU storage before acknowledging each prediction, keeping
exported tensors alive until the copy finishes. Parent process groups are untouched.
"""
from copy import copy, deepcopy
from datetime import timedelta
import os
import tempfile
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map


def _broadcast_packet(packet, groups, device, *, copy_inputs=False):
    if groups.rank == 0:
        leaves, spec = tree_flatten(packet)
        metadata = [(('tensor', tuple(v.shape), v.dtype) if isinstance(v, torch.Tensor)
                     else ('value', v)) for v in leaves]
        descriptor = [(spec, metadata)]
    else:
        leaves, descriptor = [], [None]
    dist.broadcast_object_list(descriptor, src=0, group=groups.control)
    spec, metadata = descriptor[0]
    received = []
    for i, entry in enumerate(metadata):
        if entry[0] == 'value':
            received.append(entry[1])
        else:
            value = leaves[i].to(device, copy=copy_inputs).contiguous() if groups.rank == 0 else torch.empty(entry[1], dtype=entry[2], device=device)
            dist.broadcast(value, src=0)
            received.append(value)
    return tree_unflatten(received, spec)


def _worker(rank, plan, args, model_spec, connection, rendezvous):
    try:
        device = plan['devices'][rank]
        torch.cuda.set_device(device)
        from utils.determinism import enable_deterministic_mode
        enable_deterministic_mode()
        from config.server_args import set_global_server_args
        args.device = str(device)
        from memory.allocator import configure_cuda_allocator
        configure_cuda_allocator(args.cuda_memory_limit_gib, device)
        from layers.operator_fusion.registry import resolve_operator_fusion_decision
        args.operator_fusion_decision = resolve_operator_fusion_decision(args)
        set_global_server_args(args)
        dist.init_process_group('nccl', init_method=rendezvous, rank=rank,
                                world_size=plan['topology'].world_size, timeout=timedelta(seconds=90))
        from distributed.dit_groups import DiTGroups
        groups = DiTGroups(plan['topology'])
        # Materialize the exact owner's weights, including component overrides;
        # never reload a different checkpoint behind an injected model's back.
        with torch.device('meta'):
            model = model_spec['class'].from_config(model_spec['config'])
        model.to(dtype=model_spec['dtype'])
        model.load_state_dict(model_spec['state'], assign=True, strict=True)
        if any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
            raise ValueError('unmaterialized DiT buffer in worker model')
        model.addition_config = model_spec['addition_config']
        model.eval().requires_grad_(False)
        if args.transformer_quantization != 'none':
            from models.dits.eraserdit_quantization import quantize_transformer
            quantize_transformer(model, args.pipeline_config.quantization_scope,
                                 execution_device=device, mode=args.transformer_quantization)
        tp_group, tp_ranks, tp_rank = groups.get('tp')
        if len(tp_ranks) > 1:
            from layers.dit_tensor_parallel import shard_linear_weights
            mode = args.pipeline_config.tp_linear_mode
            # Splitting N changes BF16 accumulation in the short text input
            # projection, and in the narrow output projection when SP also
            # reduces M. Replicate these small boundary layers; the remaining
            # TP weights and computation stay genuinely partitioned.
            replicated = ('caption_projection.linear_1', 'proj_out') if mode == 'aligned' else ()
            # Joint M/N splitting also changes the short-window FFN down
            # projection. Keep its GEMM N, but retain only local real weights.
            reference = tuple(f'transformer_blocks.{i}.ff.net.2' for i in range(len(model.transformer_blocks))) \
                if mode == 'aligned' and plan['topology'].sp > 1 else ()
            shard_linear_weights(model, tp_group, tp_rank, len(tp_ranks),
                                 'sharded' if mode == 'aligned' else mode, replicated_names=replicated,
                                 reference_names=reference)
        config = args.pipeline_config
        idle_residency = None
        if (rank == 0 and args.cuda_memory_limit_gib is not None
                and plan['topology'].tp == 1 and plan['topology'].replicas == 1
                and config.dit_fsdp_shard_degree * config.dit_fsdp_replicate_degree == 1):
            # The owner runs VAE on this same physical GPU. A per-process cap
            # alone cannot bound their SUM. Retain CPU checkpoint storage and
            # materialize rank 0 only while it actually executes DiT windows.
            from memory.backends.dit_idle_residency import DiTIdleResidency
            idle_residency = DiTIdleResidency(model, device)
        else:
            model.to(device)
        if config.dit_fsdp_shard_degree * config.dit_fsdp_replicate_degree > 1:
            from torch.distributed.device_mesh import init_device_mesh
            from memory.backends.fsdp_offload import shard_model, MixedPrecisionPolicy
            if config.dit_fsdp_replicate_degree > 1:
                mesh = init_device_mesh('cuda', (config.dit_fsdp_replicate_degree, config.dit_fsdp_shard_degree),
                                        mesh_dim_names=('replicate', 'shard'))
            else:
                # A real 1D mesh reuses the default group for full-world FSDP;
                # a (1, shard) mesh would allocate a redundant communicator.
                mesh = init_device_mesh('cuda', (config.dit_fsdp_shard_degree,), mesh_dim_names=('shard',))
            shard_model(model, cpu_offload=False, mesh=mesh,
                        mp_policy=MixedPrecisionPolicy(), fsdp_shard_conditions=[
                            lambda name, module: name.startswith('transformer_blocks.') and name.count('.') == 1])
        from models.adapters.eraserdit.nccl_runner import DiTRankRunner
        runner = DiTRankRunner(model, groups, config)
        from utils.dit_profile import DiTStepProfiler
        profiler = DiTStepProfiler(rank, device)
        if runner.sequence is not None:
            runner.sequence.profiler = profiler
        gpu_timings = []
        cuda_ipc = plan.get('boundary_transport') == 'cuda_ipc'
        exported_output = None
        connection.send(('ready', runner.report()))
        while True:
            # Wait on each rank's local pipe, not a collective: an idle
            # resident service must not expire the process-group timeout.
            message = connection.recv()
            command = message[0]
            # The owner sends another command only after copying and releasing
            # the previous imported output. Keep producer storage until then.
            exported_output = None
            if command == 'close':
                break
            if command == 'reset':
                runner.reset()
                profiler.reset()
                gpu_timings.clear()
                if idle_residency is not None:
                    idle_residency.release()
                from memory.allocator import release_idle_cuda_cache
                release_idle_cuda_cache(args.cuda_memory_limit_gib, device)
                torch.cuda.reset_peak_memory_stats(device)
                dist.barrier(group=groups.control)
                if rank == 0:
                    connection.send(('ok', None))
            elif command == 'predict':
                if idle_residency is not None:
                    idle_residency.acquire()
                events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                events[0].record()
                packet = _broadcast_packet(message[1] if rank == 0 else None, groups, device,
                                           copy_inputs=cuda_ipc)
                # Static conditions must not retain imported owner storage.
                del message
                events[1].record()
                with profiler.capture(model):
                    outputs = runner.predict(packet, owner_only=True)
                if rank == 0:
                    guidance_scale = packet.get('guidance_scale')
                    if guidance_scale is not None:
                        negative, positive = outputs
                        # Keep the owner's FP32 operation order, without a
                        # fused add or reduced-precision intermediate.
                        outputs = negative + guidance_scale * (positive - negative)
                        del negative, positive
                events[2].record()
                gpu_timings.append(events)
                if rank == 0:
                    if cuda_ipc:
                        exported_output = outputs
                        connection.send(('ok', exported_output))
                    else:
                        connection.send(('ok', tree_map(lambda value: value.cpu(), outputs)))
                del packet, outputs
            elif command == 'report':
                if gpu_timings:
                    gpu_timings[-1][-1].synchronize()
                report = runner.report()
                report['weight_residency'] = idle_residency.report() if idle_residency else {'mode': 'resident'}
                report['diagnostic_profiles'] = list(profiler.artifacts)
                report['window_gpu_seconds'] = {
                    'input_broadcast': sum(e[0].elapsed_time(e[1]) for e in gpu_timings) / 1000,
                    'forward_and_output_gather': sum(e[1].elapsed_time(e[2]) for e in gpu_timings) / 1000,
                }
                reports = [None] * plan['topology'].world_size
                dist.all_gather_object(reports, report, group=groups.control)
                if rank == 0:
                    connection.send(('ok', reports))
            else:
                raise ValueError(f'unknown worker command: {command}')
        dist.destroy_process_group()
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


class DiTProcessPool:
    def __init__(self, source, plan, args):
        if dist.is_initialized() and dist.get_world_size() != 1:
            raise ValueError('NCCL DiT self-launch requires a single pipeline owner process')
        if next(source.parameters()).dtype != torch.bfloat16:
            raise ValueError('NCCL DiT workers require bf16 source weights')
        self.boundary_transport = os.environ.get('MGERASE_DIT_BOUNDARY_TRANSPORT', 'cpu')
        if self.boundary_transport not in ('cpu', 'cuda_ipc'):
            raise ValueError('MGERASE_DIT_BOUNDARY_TRANSPORT must be cpu or cuda_ipc')
        plan = dict(plan, boundary_transport=self.boundary_transport)
        self.plan = plan
        self.models, self.compiled_forwards = [], []
        self.processes, self.connections = [], []
        self.closed = False
        self.store = tempfile.TemporaryDirectory(prefix='eraserdit-dit-')
        started = time.perf_counter()
        # The owner does no DiT compute. Release its duplicate GPU weights;
        # keep the module identity/configuration used by the pipeline intact.
        source.to('cpu')
        from memory.allocator import release_idle_cuda_cache
        release_idle_cuda_cache(args.cuda_memory_limit_gib, args.device)
        model_spec = {'class': type(source), 'config': dict(source.config),
                      'dtype': next(source.parameters()).dtype, 'state': source.state_dict(),
                      'addition_config': getattr(source, 'addition_config', {})}
        context = mp.get_context('spawn')
        try:
            for rank in range(plan['topology'].world_size):
                parent, child = context.Pipe()
                worker_args = copy(args)
                for name in ('distributed_context', 'parallel_context', 'official_parallel_context'):
                    setattr(worker_args, name, None)
                worker_args = deepcopy(worker_args)
                process = context.Process(target=_worker, args=(rank, plan, worker_args, model_spec, child,
                    f'file://{self.store.name}/store'), name=f'eraserdit-dit-{rank}')
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            self.initial_reports = [self._receive(rank, expected='ready') for rank in range(len(self.processes))]
        except BaseException:
            self.close(force=True)
            raise
        self.setup_seconds = time.perf_counter() - started

    def _receive(self, rank=0, *, expected='ok', timeout=300):
        def receive(index):
            try:
                return self.connections[index].recv()
            except (EOFError, OSError) as error:
                raise RuntimeError(f'NCCL DiT rank {index} IPC connection closed') from error
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.connections[rank].poll(.1):
                status, result = receive(rank)
                if status != expected:
                    raise RuntimeError(f'NCCL DiT rank {rank}: {status}: {result}')
                return result
            for index, process in enumerate(self.processes):
                # Ready messages on peer pipes must be retained until consumed.
                if not process.is_alive():
                    detail = receive(index) if self.connections[index].poll() else process.exitcode
                    raise RuntimeError(f'NCCL DiT rank {index} exited: {detail}')
                if expected == 'ok' and index and self.connections[index].poll():
                    raise RuntimeError(f'NCCL DiT rank {index} failed: {receive(index)}')
        raise TimeoutError(f'NCCL DiT rank {rank} response timed out')

    def call(self, command, payload=None):
        if self.closed:
            raise RuntimeError('NCCL DiT process pool is closed')
        try:
            if command == 'predict' and self.boundary_transport == 'cuda_ipc':
                # Inputs can be produced on a caller's non-default stream.
                devices = {v.device for v in tree_flatten(payload)[0]
                           if isinstance(v, torch.Tensor) and v.is_cuda}
                for device in devices:
                    if device != self.plan['devices'][0]:
                        raise ValueError('CUDA IPC boundary requires inputs on the rank-0 device')
                    torch.cuda.synchronize(device)
            for rank, connection in enumerate(self.connections):
                connection.send((command, payload if rank == 0 else None))
            received = self._receive()
            if command == 'predict' and self.boundary_transport == 'cuda_ipc':
                # Never expose imported worker storage to a caller. This also
                # makes held outputs valid after reset, worker failure or close.
                try:
                    result = tree_map(lambda value: value.clone(), received)
                    torch.cuda.synchronize(self.plan['devices'][0])
                finally:
                    # Drop imported handles before any exception closes workers.
                    received = None
                return result
            return received
        except (OSError, EOFError) as error:
            self.close(force=True)
            raise RuntimeError('NCCL DiT worker IPC failed') from error
        except BaseException:
            self.close(force=True)
            raise

    def close(self, force=False):
        if self.closed:
            return
        self.closed = True
        if self.connections and not force:
            try:
                for connection in self.connections:
                    connection.send(('close', None))
            except (OSError, EOFError):
                force = True
        for process in self.processes:
            if force and process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        for connection in self.connections:
            connection.close()
        self.store.cleanup()


class DiTProcessWindow:
    def __init__(self, transformer, plan, *, pool, batch=None, total_steps=0):
        self.pool, self.plan = pool, plan
        self.batch = batch
        self.rank_reports = None
        self.steps = 0
        from config.eraserdit_cache import CACHE_DEFAULTS
        self.cache_options = {key: getattr(batch, key, default) for key, default in CACHE_DEFAULTS.items()}
        self.total_steps = total_steps
        text_setting = self.cache_options['cache_text_projections']
        self.cache_text_projections = (self.cache_options['transformer_cache_mode'] != 'off'
                                       if text_setting is None else bool(text_setting))
        self.active = True
        self.boundary_seconds = dict(input_to_cpu=0., input_gpu_prepare=0., worker_roundtrip=0., output_to_device=0.)
        self.boundary_bytes = dict(input=0, output=0)

    def __enter__(self):
        self.pool.call('reset')
        return self

    def predict(self, negative, positive):
        return self._predict(negative, positive)

    def predict_guided(self, negative, positive, guidance_scale):
        """Return one FP32 guided prediction, reducing boundary output by half."""
        return self._predict(negative, positive, guidance_scale=float(guidance_scale))

    def _predict(self, negative, positive, guidance_scale=None):
        static = None
        if self.steps == 0:
            static = {name: {key: value for key, value in values.items()
                            if key not in ('hidden_states', 'timestep', 'image_rotary_emb')}
                      for name, values in (('negative', negative), ('positive', positive))}
        packet = dict(static=static, hidden=positive['hidden_states'], timestep=positive['timestep'],
                      cache_text_projections=self.cache_text_projections,
                      residual_cache=self.cache_options, total_steps=self.total_steps, step=self.steps)
        if guidance_scale is not None:
            packet['guidance_scale'] = guidance_scale
        started = time.perf_counter()
        cuda_ipc = self.pool.boundary_transport == 'cuda_ipc'
        packet = tree_map(lambda value: (value.detach() if cuda_ipc else value.detach().cpu())
                          if isinstance(value, torch.Tensor) else value, packet)
        self.boundary_seconds['input_gpu_prepare' if cuda_ipc else 'input_to_cpu'] += time.perf_counter() - started
        self.boundary_bytes['input'] += sum(v.numel() * v.element_size() for v in tree_flatten(packet)[0]
                                            if isinstance(v, torch.Tensor))
        started = time.perf_counter()
        outputs = self.pool.call('predict', packet)
        self.boundary_seconds['worker_roundtrip'] += time.perf_counter() - started
        self.boundary_bytes['output'] += sum(v.numel() * v.element_size() for v in tree_flatten(outputs)[0])
        device = positive['hidden_states'].device
        started = time.perf_counter()
        result = tree_map(lambda value: value.to(device), outputs)
        self.boundary_seconds['output_to_device'] += time.perf_counter() - started
        self.steps += 1
        return result

    def report(self):
        topology = self.plan['topology']
        self.rank_reports = self.pool.call('report')
        return dict(transport='nccl', boundary_transport=('cuda_tensor_ipc' if self.pool.boundary_transport == 'cuda_ipc'
                                                        else 'cpu_tensor_ipc'), steps=self.steps,
                    cfg_degree=topology.cfg, sp_degree=topology.sp, tp_degree=topology.tp,
                    ulysses_degree=topology.ulysses, ring_degree=topology.ring,
                    ring_attention_mode=self.plan['ring_attention_mode'],
                    output_assembly='owner_only',
                    cache_text_projections=self.cache_text_projections,
                    boundary_wall_seconds=dict(self.boundary_seconds), boundary_tensor_bytes=dict(self.boundary_bytes),
                    worker_setup_seconds=self.pool.setup_seconds, rank_reports=self.rank_reports)

    def __exit__(self, *exc):
        if not self.pool.closed:
            self.pool.call('reset')
            if self.batch is not None and self.rank_reports and exc[0] is None:
                reports = [r.get('residual_cache') for r in self.rank_reports]
                if all(reports):
                    from cache.eraserdit import aggregate_rank_cache_reports
                    for report in reports:
                        report.update(closed=True, aborted=False)
                    self.batch.extra['transformer_cache'] = aggregate_rank_cache_reports(
                        reports, sp_degree=self.plan['topology'].sp)
        return False
