# Mask STA CUDA sources

The attention kernel and the headers in `vendor/sparge` derive from
[SpargeAttn](https://github.com/thu-ml/SpargeAttn), commit
`ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a`.
The V preparation/quantization kernels and the headers in `vendor/sage` derive from
[SageAttention](https://github.com/thu-ml/SageAttention), commit
`eb615cf6cf4d221338033340ee2de1c37fbdba4a`.

Both upstream sources use Apache-2.0. Copyright notices remain in the sources
and the license text is in `vendor/LICENSE`.

EraserDiT modifications fix the validated SM89 shape, share the K/V block table
across heads, index V input and Attention output through a token permutation,
and expose a small CUDA-stream-aware C ABI, including explicit-stream V quantization.
Only the selected 128-thread
configuration is built. Runtime block tables come from each input window's mask.
