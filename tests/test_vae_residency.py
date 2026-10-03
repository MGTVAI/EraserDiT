"""VAE component leases: residency, immutable backing and mutation recovery."""
import unittest
from unittest.mock import patch

import torch
from torch import nn

from memory.backends.vae_residency import VAEPhaseResidency
from memory.tensor_ops import module_device


class ToyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 4))
        self.decoder = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 4))
        self.register_buffer('latents_mean', torch.zeros(4))
        self.register_buffer('latents_std', torch.ones(4))


class VAEResidencyTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_tied_parameters_and_replaced_buffer(self):
        model = ToyVAE().eval()
        model.decoder[0].weight = model.encoder[0].weight
        manager = VAEPhaseResidency(model, 'cuda:0', mode='cached')
        manager.acquire('vae_decode')
        self.assertTrue(model.decoder[0].weight.is_cuda)
        self.assertIs(model.decoder[0].weight, model.encoder[0].weight)
        model.latents_mean = torch.full((4,), 5., device='cuda')
        model.decoder[2].weight = nn.Parameter(torch.ones_like(model.decoder[2].weight))
        manager.release()
        self.assertTrue(all(t.device.type == 'cpu' for t in model.state_dict().values()))
        torch.testing.assert_close(model.latents_mean, torch.full((4,), 5.))
        torch.testing.assert_close(model.decoder[2].weight, torch.ones_like(model.decoder[2].weight))
        self.assertIs(model.decoder[0].weight, model.encoder[0].weight)

    def test_contracts_and_cpu_roundtrip(self):
        model = ToyVAE()
        manager = VAEPhaseResidency(model, 'cpu', mode='cached')
        with self.assertRaisesRegex(ValueError, 'eval mode'):
            manager.acquire('vae_encode')
        model.eval()
        with self.assertRaisesRegex(ValueError, 'explicit'):
            manager.acquire(None)
        manager.acquire('vae_encode')
        with self.assertRaisesRegex(RuntimeError, 'already active'):
            manager.acquire('vae_decode')
        manager.release()
        manager.release()
        self.assertFalse(hasattr(model, '_mgerase_execution_device'))

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_cuda_placement_reuse_and_updates(self):
        for mode in ('split', 'cached'):
            for pinned in (False, True):
                with self.subTest(mode=mode, pinned=pinned):
                    model = ToyVAE().eval()
                    manager = VAEPhaseResidency(model, 'cuda:0', mode=mode, pin_memory=pinned)
                    reference = ToyVAE().eval().cuda()
                    reference.load_state_dict(model.state_dict())
                    value = torch.randn(2, 4, device='cuda')
                    pointers = {}
                    for _ in range(3):
                        for phase, active, inactive in [('vae_encode', 'encoder', 'decoder'),
                                                       ('vae_decode', 'decoder', 'encoder')]:
                            manager.acquire(phase)
                            self.assertEqual(module_device(model), torch.device('cuda:0'))
                            self.assertTrue(all(p.is_cuda for p in getattr(model, active).parameters()))
                            self.assertTrue(all(p.device.type == 'cpu' for p in getattr(model, inactive).parameters()))
                            self.assertTrue(model.latents_mean.is_cuda)
                            with torch.no_grad():
                                torch.testing.assert_close(getattr(model, active)(value),
                                                           getattr(reference, active)(value), rtol=0, atol=0)
                            manager.release()
                            self.assertTrue(all(t.device.type == 'cpu' for t in model.state_dict().values()))
                            if mode == 'cached':
                                self.assertEqual(manager.stats['d2h_bytes'], 2 * 4 * 4)
                                for name, param in getattr(model, active).named_parameters():
                                    key = active + name
                                    self.assertEqual(param.is_pinned(), pinned)
                                    if key in pointers:
                                        self.assertEqual(param.data_ptr(), pointers[key])
                                    pointers[key] = param.data_ptr()
                    # CPU-side updates, then registered buffer and parameter
                    # updates while resident, must survive release and reuse.
                    with torch.no_grad():
                        model.encoder[0].weight.add_(1)
                    manager.acquire('vae_encode')
                    with torch.no_grad():
                        model.latents_mean.add_(3)
                        model.encoder[0].weight.add_(2)
                    expected = model.encoder[0].weight.detach().cpu().clone()
                    manager.release()
                    torch.testing.assert_close(model.encoder[0].weight, expected, rtol=0, atol=0)
                    torch.testing.assert_close(model.latents_mean, torch.full((4,), 3.))
                    manager.acquire('vae_encode')
                    torch.testing.assert_close(model.encoder[0].weight.cpu(), expected, rtol=0, atol=0)
                    manager.release()

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_failed_acquire_restores_and_retries(self):
        for mode in ('split', 'cached'):
            model = ToyVAE().eval()
            saved = {k: v.clone() for k, v in model.state_dict().items()}
            manager = VAEPhaseResidency(model, 'cuda:0', mode=mode)
            upload = manager._upload
            calls = 0
            def fail(tensor):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError('injected transfer failure')
                return upload(tensor)
            with patch.object(manager, '_upload', side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    manager.acquire('vae_encode')
            self.assertFalse(manager.active)
            self.assertFalse(hasattr(model, '_mgerase_execution_device'))
            for key, value in model.state_dict().items():
                self.assertEqual(value.device.type, 'cpu')
                torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
            manager.acquire('vae_decode')
            manager.release()
