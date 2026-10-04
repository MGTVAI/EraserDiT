import unittest
from unittest.mock import patch
import torch
from torch import nn

from memory.backends.dit_idle_residency import DiTIdleResidency


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
class DiTIdleResidencyTests(unittest.TestCase):
    def test_cpu_snapshot_buffers_repeated_windows_and_failure(self):
        torch.manual_seed(73)
        model = nn.Linear(32, 32).bfloat16().eval()
        model.register_buffer('scale', torch.tensor(.5))
        state = {k: v.clone() for k, v in model.state_dict().items()}
        residency = DiTIdleResidency(model, 'cuda:0')
        x = torch.randn(17, 32, device='cuda', dtype=torch.bfloat16)
        with torch.inference_mode():
            for _ in range(3):
                residency.acquire()
                actual = model(x)*model.scale
                expected = torch.nn.functional.linear(x, state['weight'].cuda(), state['bias'].cuda())*.5
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                residency.acquire()  # No redundant transfer while active.
                residency.release()
                self.assertFalse(residency.active)
                for name, value in model.state_dict().items():
                    self.assertEqual(value.device.type, 'cpu')
                    torch.testing.assert_close(value, state[name], rtol=0, atol=0)
            self.assertEqual(residency.report()['acquisitions'], 3)
            def partial_failure(*args, **kwargs):
                model.weight.data = model.weight.cuda()
                raise RuntimeError('injected transfer failure')
            with patch.object(model, 'to', side_effect=partial_failure):
                with self.assertRaisesRegex(RuntimeError, 'transfer failure'):
                    residency.acquire()
            self.assertTrue(all(t.device.type == 'cpu' for t in model.state_dict().values()))
            residency.acquire()
            torch.testing.assert_close(model(x)*model.scale, expected, rtol=0, atol=0)
            residency.release()
