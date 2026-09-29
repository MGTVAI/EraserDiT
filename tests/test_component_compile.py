"""Auxiliary graph capture, lifecycle, configuration and device transfers."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from config.server_args import ServerArgs
from layers.component_compile import ComponentCompileManager


class ComponentCompileTests(unittest.TestCase):
    def test_contracts_and_independence(self):
        args = ServerArgs(compile_components='vae_encoder,vae_decoder,vae_encoder', vae_cpu_offload=True)
        self.assertEqual(args.compile_components, ('vae_encoder','vae_decoder'))
        self.assertFalse(args.enable_torch_compile)
        with self.assertRaisesRegex(ValueError, 'FSDP'):
            ServerArgs(compile_components='text_encoder', text_encoder_cpu_offload=True)
        with self.assertRaisesRegex(ValueError, 'single-GPU VAE'):
            ServerArgs(compile_components='vae_decoder', pipeline_config=SimpleNamespace(vae_degree=2))
        with self.assertRaises(ValueError):
            ServerArgs(compile_components='unknown')
        ServerArgs(compile_components='vae_decoder', pipeline_config=SimpleNamespace(cfg_degree=2,vae_degree=1))

    def test_rollback_close_and_metadata(self):
        model = torch.nn.Linear(3, 4).eval()
        eager = model.forward
        with patch('layers.component_compile.torch.compile', side_effect=lambda f, **kw: f):
            with self.assertRaises(KeyError):
                ComponentCompileManager({'text_encoder':model}, ('text_encoder','vae_encoder'), mode='default')
            self.assertEqual(model.forward, eager)
            manager = ComponentCompileManager({'text_encoder':model}, ('text_encoder',), mode='default')
        keys = list(model.state_dict())
        with torch.no_grad():
            model(torch.ones(1, 3))
            saved = manager.snapshot()
            model(torch.zeros(1, 3))
        self.assertEqual(saved['text_encoder']['successful_forwards'],1)
        self.assertEqual(manager.snapshot()['text_encoder']['successful_forwards'],2)
        self.assertEqual(list(model.state_dict()),keys)
        with self.assertRaisesRegex(RuntimeError,'inference-only'):
            model(torch.ones(1,3))
        manager.close(); manager.close()
        self.assertEqual(model.forward,eager)

    def test_real_t5_fullgraph(self):
        from transformers import T5Config, T5EncoderModel
        model = T5EncoderModel(T5Config(vocab_size=64,d_model=32,d_ff=64,num_layers=2,num_heads=2,d_kv=16,dropout_rate=0.)).eval()
        graphs=[]
        real_compile=torch.compile
        def backend(gm, inputs):
            graphs.append(gm)
            return gm.forward
        with patch('layers.component_compile.torch.compile',side_effect=lambda f,**kw: real_compile(f,backend=backend,fullgraph=True,dynamic=False)):
            manager=ComponentCompileManager({'text_encoder':model}, ('text_encoder',), mode='default')
        self.addCleanup(manager.close)
        self.addCleanup(torch._dynamo.reset)
        eager=manager.entries[0][1].eager
        with torch.no_grad():
            for length in (8,16,8):
                ids=torch.arange(length).unsqueeze(0)
                torch.testing.assert_close(model(ids)[0],eager(ids)[0])
        self.assertEqual(len(graphs),2)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get('ERASERDIT_TEST_COMPONENT_COMPILE')=='1', 'opt-in CUDA required')
    def test_vae_device_roundtrip_and_actual_inductor(self):
        from models.vaes.eraserdit_vae import LTXVideoCausalConv3d
        encoder=LTXVideoCausalConv3d(3,4).cuda().bfloat16().eval()
        decoder=LTXVideoCausalConv3d(4,3).cuda().bfloat16().eval()
        vae=torch.nn.Module(); vae.encoder=encoder; vae.decoder=decoder
        manager=ComponentCompileManager({'vae':vae},('vae_encoder','vae_decoder'),mode='default')
        self.addCleanup(manager.close)
        with torch.no_grad():
            for frames in (3,5,3):
                vae.cpu(); vae.cuda()
                x=torch.randn(1,3,frames,8,8,device='cuda',dtype=torch.bfloat16)
                actual=vae.decoder(vae.encoder(x))
                expected=manager.entries[1][1].eager(manager.entries[0][1].eager(x))
                torch.testing.assert_close(actual,expected,atol=0.02,rtol=0.02)
        self.assertEqual(manager.snapshot()['vae_encoder']['successful_forwards'],3)


if __name__=='__main__':
    unittest.main()
