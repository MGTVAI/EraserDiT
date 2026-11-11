"""Window-owned mask-protected STA for the validated SM89 Q6c layout."""
from contextlib import contextmanager
from contextvars import ContextVar
import ctypes
from functools import lru_cache
import hashlib
from itertools import product
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from layers.audited_allocations import empty, empty_like


GRID = (16, 34, 60)
TILE = (4, 2, 4)
TILE_GRID = (4, 17, 15)
BACKGROUND_WINDOW = (4, 11, 7)
PROTECT_KERNEL = (3, 5, 5)
LENGTH, HEADS, DIM = 32640, 32, 64
QUERY_BLOCK, KV_BLOCK = 128, 64
_active_window = ContextVar('eraserdit_mask_sta_window', default=None)


@triton.jit
def _quant_query(X, ORDER, Y, S, L: tl.constexpr, H: tl.constexpr, D: tl.constexpr):
    program = tl.program_id(0)
    head = tl.program_id(1)
    row = (program // 8) * 32 + tl.arange(0, 4) * 8 + program % 8
    source = tl.load(ORDER + row)
    col = head * D + tl.arange(0, D)
    value = tl.load(X + source[:, None] * (H * D) + col[None, :]).to(tl.float32)
    scale = tl.max(tl.abs(value)) / 127. + 0.0000001
    quantized = value / scale
    quantized += 0.5 * tl.where(quantized >= 0, 1, -1)
    tl.store(Y + row[:, None] * (H * D) + col[None, :], quantized.to(tl.int8))
    tl.store(S + head * (L // 32 * 8) + program, scale)


@triton.jit
def _quant_key(X, ORDER, M, Y, S, L: tl.constexpr, H: tl.constexpr, D: tl.constexpr):
    program = tl.program_id(0)
    head = tl.program_id(1)
    rows = (program // 4) * 64 + tl.arange(0, 8) * 8 + (program % 4) * 2
    source0 = tl.load(ORDER + rows)
    source1 = tl.load(ORDER + rows + 1)
    col = head * D + tl.arange(0, D)
    mean = tl.load(M + col).to(tl.float32)
    value0 = tl.load(X + source0[:, None] * (H * D) + col[None, :]).to(tl.float32)
    value1 = tl.load(X + source1[:, None] * (H * D) + col[None, :]).to(tl.float32)
    value0 = (value0 - mean[None, :]).to(tl.bfloat16).to(tl.float32)
    value1 = (value1 - mean[None, :]).to(tl.bfloat16).to(tl.float32)
    maximum = tl.maximum(tl.max(tl.max(tl.abs(value0), 1), 0), tl.max(tl.max(tl.abs(value1), 1), 0))
    scale = maximum / 127. + 0.0000001
    quantized0, quantized1 = value0 / scale, value1 / scale
    quantized0 += 0.5 * tl.where(quantized0 >= 0, 1, -1)
    quantized1 += 0.5 * tl.where(quantized1 >= 0, 1, -1)
    tl.store(Y + rows[:, None] * (H * D) + col[None, :], quantized0.to(tl.int8))
    tl.store(Y + (rows[:, None] + 1) * (H * D) + col[None, :], quantized1.to(tl.int8))
    tl.store(S + head * (L // 64 * 4) + program, scale)


def block_geometry(device):
    def morton(y, x):
        return sum((((x >> i) & 1) << (2*i)) + (((y >> i) & 1) << (2*i+1)) for i in range(5))

    spatial = sorted(product(range(TILE_GRID[1]), range(TILE_GRID[2])), key=lambda hw: morton(*hw))
    tiles = [(t, h, w) for t in range(TILE_GRID[0]) for h, w in spatial]
    offsets = list(product(*(range(axis) for axis in TILE)))
    order = [((t*TILE[0]+dt)*GRID[1] + h*TILE[1]+dh)*GRID[2] + w*TILE[2]+dw
             for t, h, w in tiles for dt, dh, dw in offsets]
    permutation = torch.tensor(order, dtype=torch.long, device=device)
    coordinates = torch.tensor(tiles, device=device)
    center = coordinates.reshape(LENGTH//QUERY_BLOCK, 4, 3).sum(1)//4
    extent = torch.tensor(BACKGROUND_WINDOW, device=device)
    start = torch.minimum((center - extent//2).clamp_min(0),
                          torch.tensor(TILE_GRID, device=device) - extent)
    selected = ((coordinates[None] >= start[:, None])
                & (coordinates[None] < (start+extent)[:, None])).all(-1)
    background = selected.reshape(LENGTH//QUERY_BLOCK, LENGTH//KV_BLOCK, 2).any(-1)
    query = torch.arange(LENGTH//QUERY_BLOCK, device=device)
    background[query, query*2] = True
    background[query, query*2+1] = True
    return permutation, background


def protected_blocks(mask, permutation):
    if tuple(mask.shape) != (1, 1, *GRID):
        raise ValueError(f'mask STA requires mask [1,1,{GRID[0]},{GRID[1]},{GRID[2]}]')
    binary = mask > 0
    expanded = F.max_pool3d(binary.float(), PROTECT_KERNEL, stride=1, padding=(1, 2, 2)) > 0
    query = expanded.flatten().index_select(0, permutation).view(-1, QUERY_BLOCK).any(1)
    return query, binary, expanded


def block_table(mask):
    columns = torch.arange(mask.shape[1], device=mask.device)
    positions = torch.where(mask, columns, mask.shape[1]).sort(dim=-1).values
    counts = mask.sum(-1, dtype=torch.int32)
    positions = torch.where(columns < counts[:, None], positions, 0).to(torch.int32)
    previous = F.pad(positions[:, :-1], (1, 0))
    return (positions-previous).contiguous(), counts.contiguous()


@lru_cache(maxsize=4)
def _native_library(path, expected_sha256):
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise RuntimeError('mask STA library SHA256 differs from the recorded build')
    library = ctypes.CDLL(str(path))
    library.eraserdit_mask_sta_abi.argtypes = []
    library.eraserdit_mask_sta_abi.restype = ctypes.c_int
    if library.eraserdit_mask_sta_abi() != 2:
        raise RuntimeError('unsupported mask STA library ABI')
    pointer = ctypes.c_void_p
    library.sta_mask_attention.argtypes = [pointer]*10 + [ctypes.c_float, pointer]
    library.sta_mask_attention.restype = ctypes.c_int
    library.sta_mask_prepare_v.argtypes = [pointer]*4
    library.sta_mask_prepare_v.restype = ctypes.c_int
    library.sta_mask_quantize_v.argtypes = [pointer]*4
    library.sta_mask_quantize_v.restype = ctypes.c_int
    return library, str(path), digest


def _check(status, operation):
    if status:
        raise RuntimeError(f'mask STA {operation} failed with CUDA status {status}')


class MaskSTA:
    def __init__(self, mask, grid, library_path, library_sha256=''):
        if tuple(grid) != GRID or tuple(mask.shape) != (1, 1, *GRID):
            raise ValueError(f'mask STA supports only token grid {GRID}, batch 1')
        if mask.device.type != 'cuda' or torch.cuda.get_device_capability(mask.device) != (8, 9):
            raise ValueError('mask STA requires a single SM89 CUDA worker')
        self.native, path, digest = _native_library(library_path, library_sha256)
        self.device = mask.device
        self.setup_stream = torch.cuda.current_stream(self.device)
        self.permutation, background = block_geometry(self.device)
        self.row_order = self.permutation.to(torch.int32)
        protected, binary, expanded = protected_blocks(mask, self.permutation)
        background[protected] = True
        self.lut, self.counts = block_table(background)
        self._report = dict(
            enabled=True, grid=GRID, background_window=BACKGROUND_WINDOW, protect_kernel=PROTECT_KERNEL,
            query_block=QUERY_BLOCK, kv_block=KV_BLOCK, calls=0, closed=False,
            keep_fraction=float(background.double().mean()), protected_query_blocks=int(protected.sum()),
            total_query_blocks=LENGTH//QUERY_BLOCK, mask_tokens=int(binary.sum()),
            boundary_tokens=int((expanded & ~binary).sum()), library=path, library_sha256=digest)

    def report(self):
        return dict(self._report)

    def close(self):
        self._report['closed'] = True
        self.permutation = self.row_order = self.lut = self.counts = None

    def __call__(self, q, k, v, scale, peripheral):
        if self._report['closed']:
            raise RuntimeError('mask STA window is closed')
        if (peripheral is None or not peripheral.supported(q, k, v) or q.device != self.device
                or torch.is_grad_enabled() or torch.compiler.is_compiling()):
            raise ValueError('mask STA requires eager contiguous BF16 Q/K/V [1,32640,32,64]')
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            if stream != self.setup_stream:
                stream.wait_stream(self.setup_stream)
                for tensor in (self.permutation, self.row_order, self.lut, self.counts):
                    tensor.record_stream(stream)
            mean = k.mean(dim=1, keepdim=True)
            qi = empty_like(q, dtype=torch.int8, label='sage_query_int8')
            ki = empty_like(k, dtype=torch.int8, label='sage_key_int8')
            qs = empty((1, HEADS, LENGTH//32*8), device=q.device, dtype=torch.float32, label='sage_query_scale')
            ks = empty((1, HEADS, LENGTH//64*4), device=q.device, dtype=torch.float32, label='sage_key_scale')
            _quant_query[(LENGTH//32*8, HEADS)](q, self.row_order, qi, qs, LENGTH, HEADS, DIM)
            _quant_key[(LENGTH//64*4, HEADS)](k, self.row_order, mean, ki, ks,
                                           LENGTH, HEADS, DIM, num_warps=4)
            vt = empty((1, DIM, HEADS, LENGTH), device=q.device, dtype=v.dtype, label='sage_v_transpose')
            vf = empty_like(vt, dtype=torch.float8_e4m3fn, label='sage_v_fp8')
            vs = empty((1, HEADS, DIM), device=q.device, dtype=torch.float32, label='sage_v_scale')
            _check(self.native.sta_mask_prepare_v(v.data_ptr(), vt.data_ptr(),
                   self.row_order.data_ptr(), stream.cuda_stream), 'V preparation')
            _check(self.native.sta_mask_quantize_v(vt.data_ptr(), vf.data_ptr(),
                   vs.data_ptr(), stream.cuda_stream), 'V quantization')
            output = empty_like(q, dtype=torch.bfloat16, label='sage_output')
            tensors = (qi, ki, vf, output, self.lut, self.counts, qs, ks, vs, self.row_order)
            _check(self.native.sta_mask_attention(*(tensor.data_ptr() for tensor in tensors),
                                                 scale, stream.cuda_stream), 'attention')
        self._report['calls'] += 1
        return output


def current_mask_sta():
    return _active_window.get()


@contextmanager
def mask_sta_window(mask, grid, library_path, library_sha256=''):
    operator = MaskSTA(mask, grid, library_path, library_sha256)
    token = _active_window.set(operator)
    try:
        yield operator
    finally:
        _active_window.reset(token)
        operator.close()
