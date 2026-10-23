"""Fixed-shape NHD per-thread quantization candidates; original scale groups.
Retain BF16 rounding of K-kmean before amax and quantization.
"""
import torch,triton
import triton.language as tl
@triton.jit
def _key_fused(X,M,Y,S,L:tl.constexpr,H:tl.constexpr,D:tl.constexpr,GROUPS:tl.constexpr):
    program=tl.program_id(0);head=tl.program_id(1)
    g=tl.arange(0,GROUPS);r=tl.arange(0,8);c=tl.arange(0,D)
    if GROUPS==4:block=program;group=g
    else:block=program//4;group=program%4+g
    n0=block*64+r[None,:]*8+group[:,None]*2;n1=n0+1
    mu=tl.load(M+head*D+c).to(tl.float32)
    x0=tl.load(X+n0[:,:,None]*(H*D)+head*D+c[None,None,:]).to(tl.float32)
    x1=tl.load(X+n1[:,:,None]*(H*D)+head*D+c[None,None,:]).to(tl.float32)
    x0=(x0-mu[None,None,:]).to(tl.bfloat16).to(tl.float32)
    x1=(x1-mu[None,None,:]).to(tl.bfloat16).to(tl.float32)
    maximum=tl.maximum(tl.max(tl.max(tl.abs(x0),2),1),tl.max(tl.max(tl.abs(x1),2),1))
    scale=maximum/127.+0.0000001
    z0=x0/scale[:,None,None];z1=x1/scale[:,None,None]
    z0+=0.5*tl.where(z0>=0,1,-1);z1+=0.5*tl.where(z1>=0,1,-1)
    tl.store(Y+n0[:,:,None]*(H*D)+head*D+c[None,None,:],z0.to(tl.int8))
    tl.store(Y+n1[:,:,None]*(H*D)+head*D+c[None,None,:],z1.to(tl.int8))
    tl.store(S+head*(L//64*4)+block*4+group,scale)

# The native V preparation/core are the same functions used by this Sage build.
# The adapter changes only explicit full-write allocations and K subtraction.
_CORE_SHA = 'eb5ca9e8047583cc3a139f788f4bf48857aa1134d47a36ad5021b153690beaf4'
_QUANT_SHA = '86bf415d4d24793a591c2a056e60f41b4bc579519ac7a875e484a65816940bf7'
_NATIVE_SHA = 'c3f7a13e3815c7b97c43864f2d8ef576cb76f565688573c954cf0badc9974b56'


class SagePeripheral:
    def __init__(self):
        import hashlib
        import importlib
        from pathlib import Path
        import sageattention.core as core
        import sageattention.quant as quant
        from sageattention.triton.quant_per_thread import quant_query_per_thread_int8_kernel
        native = importlib.import_module('sageattention._qattn_sm89')
        for module, expected in ((core, _CORE_SHA), (quant, _QUANT_SHA),
                (core.sm89_compile, 'dd5e0ebf92f2662a85d24cee5c984fe6ba114e610253b4bd339af5e3d5da97c9'),
                (native, _NATIVE_SHA)):
            if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Q6 Sage peripheral requires the validated SageAttention eb615cf SM89 build')
        self.core, self.quant, self.query_kernel = core, quant, quant_query_per_thread_int8_kernel
        self.calls = 0
        self.fallbacks = 0

    def supported(self, q, k, v):
        return (q.shape == k.shape == v.shape == (1, 32640, 32, 64)
                and q.dtype == k.dtype == v.dtype == torch.bfloat16
                and q.is_cuda and q.device == k.device == v.device
                and all(t.is_contiguous() for t in (q, k, v)))

    def __call__(self, q, k, v, sm_scale):
        from layers.audited_allocations import empty
        if not self.supported(q, k, v) or torch.is_grad_enabled() or torch.compiler.is_compiling():
            raise ValueError('Q6 Sage peripheral requires eager inference on contiguous BF16 [1,32640,32,64]')
        with torch.cuda.device(v.device):
            length, heads, dim = 32640, 32, 64
            km = k.mean(dim=1, keepdim=True)
            qi = empty(q.shape, device=q.device, dtype=torch.int8, label='sage_query_int8')
            ki = empty(k.shape, device=k.device, dtype=torch.int8, label='sage_key_int8')
            qs = empty((1, heads, length//32*8), device=q.device, dtype=torch.float32, label='sage_query_scale')
            ks = empty((1, heads, length//64*4), device=k.device, dtype=torch.float32, label='sage_key_scale')
            self.query_kernel[(length//32*8, heads, 1)](q, qi, qs, length,
                q.stride(0), q.stride(2), q.stride(1), qi.stride(0), qi.stride(2), qi.stride(1), qs.stride(0), qs.stride(1), C=dim, BLK=32)
            _key_fused[(length//64*4, heads)](k, km, ki, ks, length, heads, dim, 1, num_warps=4)
            output = empty(q.shape, device=q.device, dtype=q.dtype, label='sage_output')
            vt = empty((1, dim, heads, length), device=v.device, dtype=v.dtype, label='sage_v_transpose')
            vf = empty(vt.shape, device=v.device, dtype=torch.float8_e4m3fn, label='sage_v_fp8')
            vs = empty((1, heads, dim), device=v.device, dtype=torch.float32, label='sage_v_scale')
            # smooth_v=False: upstream vm is allocated but never read; omit it.
            self.quant._fused.transpose_pad_permute_cuda(v, vt, 0)
            self.quant._fused.scale_fuse_quant_cuda(vt, vf, vs, length, 448.0, 0)
            self.core.sm89_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
                qi, ki, vf, output, qs, ks, vs, 0, 0, 3, sm_scale, 0)
        self.calls += 1
        return output
