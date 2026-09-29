"""Static inference reuse must preserve values and request isolation."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from config.server_args import ServerArgs, set_global_server_args


class StaticConditionTests(unittest.TestCase):
    def test_serial_denoising_keeps_cfg_inputs_and_scheduler_order(self):
        from config.eraserdit import EraserDiTEraseSamplingParams, EraserDiTPipelineConfig
        from nodes.schedule_batch import Req
        from pipelines.stages.eraserdit_erase._common import MODEL_FRAMES_KEY
        from pipelines.stages.eraserdit_erase.denoising import EraserDiTEraseDenoisingStage

        calls, predictions = [], []
        rotary = object()

        class Transformer(torch.nn.Module):
            config = SimpleNamespace(patch_size=1, patch_size_t=1)
            transformer_blocks = []

            def rope(self, *args):
                return rotary

            def forward(self, **kwargs):
                calls.append(kwargs)
                return (kwargs['hidden_states'] + kwargs['encoder_hidden_states'].mean(),)

        def step(noise, timestep, latents, return_dict):
            predictions.append(noise.clone())
            return (latents - noise,)

        args = ServerArgs(device='cpu', pipeline_config=EraserDiTPipelineConfig())
        model = Transformer()
        stage = EraserDiTEraseDenoisingStage(model, SimpleNamespace(step=step), args)
        batch = Req(
            sampling_params=EraserDiTEraseSamplingParams(num_inference_steps=2, guidance_scale=2),
            modules={'vae': SimpleNamespace(temporal_compression_ratio=8)},
            extra={MODEL_FRAMES_KEY: 1}, latents=torch.zeros(1, 1, 1, 2, 2),
            cond_latents=torch.ones(1, 1, 1, 2, 2), mask_values=torch.ones(1, 1, 1, 2, 2),
            prompt_embeds=torch.full((1, 4, 8), 3.), negative_prompt_embeds=torch.ones(1, 4, 8),
            prompt_attention_mask=torch.ones(1, 4), negative_attention_mask=torch.zeros(1, 4),
            timesteps=torch.tensor([2., 1.]),
        )
        with patch('torch.autocast', return_value=nullcontext()):
            stage.forward(batch, args)
        self.assertEqual(len(calls), 4)
        for index, values in enumerate(calls):
            negative = index % 2 == 0
            self.assertIs(values['encoder_hidden_states'],
                          batch.negative_prompt_embeds if negative else batch.prompt_embeds)
            self.assertIs(values['encoder_attention_mask'],
                          batch.negative_attention_mask if negative else batch.prompt_attention_mask)
            self.assertIs(values['image_rotary_emb'], rotary)
            self.assertIs(values['cond_latents'], batch.cond_latents)
            self.assertEqual(values['timestep'].item(), 2 - index // 2)
        torch.testing.assert_close(predictions[0], torch.full_like(batch.latents, 5.))
        torch.testing.assert_close(predictions[1], torch.zeros_like(batch.latents))
        torch.testing.assert_close(batch.latents, torch.full_like(batch.latents, -5.))

    def test_text_cache_invalidates_and_is_request_owned(self):
        from pipelines.stages.eraserdit_erase._common import EraserDiTTaskState, TASK_STATE_KEY
        from pipelines.stages.eraserdit_erase.text_encoding import EraserDiTEraseTextEncodingStage
        args = ServerArgs(device='cpu')
        set_global_server_args(args)
        encoder = torch.nn.Linear(1, 1)
        encoder.dtype = torch.float32
        stage = EraserDiTEraseTextEncodingStage(encoder, object())
        state = EraserDiTTaskState()
        def batch(state=state, prompt='positive', length=8):
            return SimpleNamespace(extra={TASK_STATE_KEY: state}, prompt=prompt,
                                   negative_prompt='', max_sequence_length=length, metrics=None)
        def encode(prompt, *, max_sequence_length, **kwargs):
            return (torch.full((1, max_sequence_length, 4), float(len(prompt))),
                    torch.ones(1, max_sequence_length, dtype=torch.bool))
        with patch.object(stage, '_t5_prompt_embeds', side_effect=encode) as call, \
                patch('torch.autocast', return_value=nullcontext()):
            first = stage.forward(batch(), args)
            # Caller mutations must not overwrite the CPU cache.
            first.prompt_embeds.add_(1)
            second = stage.forward(batch(), args)
            self.assertEqual(call.call_count, 2)
            self.assertTrue(torch.equal(second.prompt_embeds, torch.full((1, 8, 4), 8.)))
            stage.forward(batch(prompt='changed'), args)
            self.assertEqual(call.call_count, 4)
            stage.forward(batch(prompt='changed', length=9), args)
            self.assertEqual(call.call_count, 6)
            stage.forward(batch(state=EraserDiTTaskState()), args)
            self.assertEqual(call.call_count, 8)
        self.assertTrue(all(t.device.type == 'cpu' for t in state.extra['text_encoding'][1]))

    def check_rotary_reuse(self, device):
        from models.dits.eraserdit_transformer import EraserDiTLTXVideoTransformer3DModel
        set_global_server_args(ServerArgs(device=device, attention_backend='sdpa'))
        torch.manual_seed(19)
        model = EraserDiTLTXVideoTransformer3DModel(
            in_channels=3, out_channels=1, num_attention_heads=2,
            attention_head_dim=16, cross_attention_dim=32, num_layers=2,
            caption_channels=16,
        ).to(device).eval()
        with torch.no_grad():
            for frames, height, width in ((1, 2, 3), (2, 3, 2)):
                hidden = torch.randn(1, 1, frames, height, width, device=device)
                kwargs = dict(hidden_states=hidden, cond_latents=torch.randn_like(hidden),
                              mask_values=torch.ones_like(hidden),
                              encoder_hidden_states=torch.randn(1, 4, 16, device=device),
                              encoder_attention_mask=torch.ones(1, 4, device=device),
                              timestep=torch.ones(1, device=device),
                              num_frames=frames, height=height, width=width, return_dict=False)
                reference = model(**kwargs)[0]
                rope = model.rope(hidden, frames, height, width, None)
                with patch.object(model.rope, 'forward', side_effect=AssertionError('recomputed RoPE')):
                    actual = model(**kwargs, image_rotary_emb=rope)[0]
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_rotary_reuse_cpu(self):
        self.check_rotary_reuse('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_rotary_reuse_cuda(self):
        self.check_rotary_reuse('cuda')


if __name__ == '__main__':
    unittest.main()
