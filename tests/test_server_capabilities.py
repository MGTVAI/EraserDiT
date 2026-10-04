"""Capabilities shown to clients follow the configured inference path."""
from types import SimpleNamespace
import unittest

from config.eraserdit import EraserDiTPipelineConfig
from config.server_args import ServerArgs
from entrypoints.server.serve import _effective_acceleration


class ServerCapabilitiesTests(unittest.TestCase):
    def modes(self, **kwargs):
        args = ServerArgs(**kwargs)
        report = _effective_acceleration(args, SimpleNamespace(session=SimpleNamespace(pipeline=None)))
        self.assertEqual(report['cuda_memory_limit_gib'], args.cuda_memory_limit_gib)
        return report['transformer_cache']['supported_modes']

    def test_offload_and_compile(self):
        self.assertEqual(self.modes(dit_layerwise_offload=True), ['off','teacache'])
        self.assertEqual(self.modes(dit_layerwise_offload=False,dit_cpu_offload=True,
                                   cuda_memory_limit_gib=22), ['off','teacache','cache_dit'])
        self.assertEqual(self.modes(enable_torch_compile=True), ['off'])

    def test_nccl_topology(self):
        config = EraserDiTPipelineConfig(dit_parallel_backend='nccl',cfg_degree=2)
        self.assertEqual(self.modes(pipeline_config=config,dit_layerwise_offload=False),
                         ['off','teacache','cache_dit'])
        config = EraserDiTPipelineConfig(dit_parallel_backend='nccl',sp_degree=2,
                                        sp_attention_mode='ring',ring_degree=2)
        self.assertEqual(self.modes(pipeline_config=config,dit_layerwise_offload=False), ['off'])
