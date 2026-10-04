import os
import unittest
import torch


@unittest.skipUnless(os.environ.get('ERASERDIT_TEST_INT8')=='1' and torch.cuda.is_available(),
                     'requires CUDA and ERASERDIT_TEST_INT8=1')
class Fp8AuditTests(unittest.TestCase):
    def test_statistics_before_and_after_gelu(self):
        from entrypoints.cli.audit_static_fp8 import activation_statistics
        from layers.quantization.gelu import make_gelu_lut
        x=torch.tensor([-100,-56,-1,0,56,100,float('nan'),float('inf')],device='cuda',dtype=torch.bfloat16)
        before=activation_statistics(x)
        self.assertEqual(before,dict(elements=8,max_abs=100.,clipped=2,nonfinite=2))
        lut=make_gelu_lut(x.device)
        after=activation_statistics(x[:6],lut)
        self.assertEqual(after,dict(elements=6,max_abs=100.,clipped=1,nonfinite=0))
        self.assertTrue(torch.isnan(x[-2]))
