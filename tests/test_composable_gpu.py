"""Unsupported migrated offload combinations fail before allocating replicas."""
import unittest
from unittest.mock import patch
from config.server_args import ServerArgs
from config.eraserdit import EraserDiTPipelineConfig
from memory.validation import validate_memory_config

class ComposableTests(unittest.TestCase):
    def test_cfg_sp_allow_primary_text_and_vae_offload(self):
        from models.adapters.eraserdit.mesh import resolve_mesh
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=2):
            for field in ('cfg_degree', 'sp_degree'):
                args = ServerArgs(device='cuda:0', text_encoder_cpu_offload=True, vae_cpu_offload=True,
                                  pipeline_config=EraserDiTPipelineConfig(**{field: 2}))
                validate_memory_config(args)
                self.assertEqual(len(resolve_mesh(args)['devices']), 2)
                args.transformer_quantization = 'int8_w8a8_native'
                validate_memory_config(args)

    def test_thread_mesh_with_offload_rejected(self):
        with patch('torch.cuda.is_available', return_value=True):
            for field in ('cfg_degree', 'sp_degree', 'vae_degree'):
                args = ServerArgs(device='cuda', dit_layerwise_offload=True,
                                  pipeline_config=EraserDiTPipelineConfig(**{field: 2}))
                with self.assertRaisesRegex(ValueError, 'single-GPU'):
                    validate_memory_config(args)
