"""Bound the approximate NCCL path to implementations reconstructed by workers."""
import unittest
from config.eraserdit import EraserDiTPipelineConfig
from config.server_args import ServerArgs
from config.dit_parallel import validate_nccl_dit

class NcclQuantizedConfigTests(unittest.TestCase):
    def test_static_ffn_sage_fast_cfg_sp_configs(self):
        for cfg,sp in ((2,1),(1,2),(1,4),(2,2)):
            args=ServerArgs(pipeline_config=EraserDiTPipelineConfig(
                dit_parallel_backend='nccl',cfg_degree=cfg,sp_degree=sp,
                sp_linear_mode='aligned' if sp>1 else 'reference',quantization_scope='ffn'),
                transformer_quantization='fp8_w8a8_static',attention_backend='sage_fp8',
                operator_fusion_backend='triton',
                operator_fusion_ops='qk_rmsnorm_rope_fast,rmsnorm_adaln_fast,gated_residual')
            validate_nccl_dit(args)
            from layers.operator_fusion.registry import resolve_operator_fusion_decision
            decision=resolve_operator_fusion_decision(args)
            self.assertEqual(set(decision.effective_ops),set(args.operator_fusion_ops.split(',')))

    def test_reject_unimplemented_topology_and_scope(self):
        for options in (dict(tp_degree=2,quantization_scope='ffn'),
                        dict(sp_degree=2,quantization_scope='blocks'),
                        dict(dit_fsdp_shard_degree=2,quantization_scope='ffn')):
            with self.assertRaisesRegex(ValueError,'NCCL'):
                ServerArgs(pipeline_config=EraserDiTPipelineConfig(dit_parallel_backend='nccl',**options),
                           transformer_quantization='fp8_w8a8_static',attention_backend='sage_fp8')
