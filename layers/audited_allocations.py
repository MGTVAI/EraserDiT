"""Bounded fill suppression at audited full-write call sites only.

PyTorch's flag is process-global. Opt-in is restricted to a private, main-thread
inference worker, not concurrent service requests. No torch/Sage monkeypatches.
Every individual allocation/native GEMM restores the caller's flag on failure.
"""
from collections import Counter
from contextlib import contextmanager
import threading
import torch
from config.q6_extensions import q6_extensions

_COUNTS = Counter()


@contextmanager
def _allocation_scope(label, enabled):
    skip = enabled and q6_extensions().targeted_fill
    if not skip:
        yield
        return
    if threading.current_thread() is not threading.main_thread() or torch.compiler.is_compiling() or torch.is_grad_enabled():
        raise RuntimeError('targeted fill requires standalone main-thread eager inference')
    previous = torch.utils.deterministic.fill_uninitialized_memory
    torch.utils.deterministic.fill_uninitialized_memory = False
    try:
        yield
        _COUNTS[label] += 1
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous


def empty_like(value, *, label, **kwargs):
    if torch.compiler.is_compiling():
        return torch.empty_like(value, **kwargs)
    with _allocation_scope(label, value.is_cuda):
        return torch.empty_like(value, **kwargs)


def empty(shape, *, label, **kwargs):
    if torch.compiler.is_compiling():
        return torch.empty(shape, **kwargs)
    with _allocation_scope(label, torch.device(kwargs.get('device', 'cpu')).type == 'cuda'):
        return torch.empty(shape, **kwargs)


def scaled_mm(*args, label, audited=True, **kwargs):
    if torch.compiler.is_compiling():
        return torch._scaled_mm(*args, **kwargs)
    with _allocation_scope(label, audited and args[0].is_cuda):
        return torch._scaled_mm(*args, **kwargs)


def snapshot():
    return dict(enabled=q6_extensions().targeted_fill, skipped_calls=dict(_COUNTS),
                fill_uninitialized_memory=torch.utils.deterministic.fill_uninitialized_memory,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                scope='individual audited full-write outputs; private main-thread inference worker only')
