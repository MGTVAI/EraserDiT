"""Meta loading materializes checkpoint storage and preserves aliases."""
import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn
from safetensors.torch import save_file
from loader.meta_load import load_safetensors_model

class Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(3,3)
        self.b = nn.Linear(3,3)
        self.b.weight = self.a.weight
        self.register_buffer('scale', torch.ones(3))

class MetaLoadTests(unittest.TestCase):
    def test_shards_tied_weights_and_dtype(self):
        model = Tied().eval()
        with tempfile.TemporaryDirectory() as path:
            state = {k:v.contiguous().clone() for k,v in model.state_dict().items() if k != 'b.weight'}
            save_file({'a.weight': state.pop('a.weight')}, str(Path(path)/'one.safetensors'))
            save_file(state, str(Path(path)/'two.safetensors'))
            loaded, _ = load_safetensors_model(Tied, path, dtype=torch.bfloat16)
        self.assertIs(loaded.a.weight, loaded.b.weight)
        self.assertFalse(any(t.is_meta for t in loaded.parameters()))
        for key,value in loaded.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key].to(torch.bfloat16), rtol=0,atol=0)

    def test_missing_unexpected_and_wrong_shape_fail(self):
        for state in ({'weight':torch.zeros(2,2)}, {'unknown':torch.zeros(2,2)},
                      {'weight':torch.zeros(3,3),'bias':torch.zeros(2)}):
            with tempfile.TemporaryDirectory() as path:
                save_file(state,str(Path(path)/'weights.safetensors'))
                with self.assertRaises(ValueError):
                    load_safetensors_model(lambda:nn.Linear(2,2),path,dtype=torch.float32)
