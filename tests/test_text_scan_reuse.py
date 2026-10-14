import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from pipelines.stages.eraserdit_erase import text_encoding as text
from utils import video_io as video

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.eval()
    @property
    def dtype(self): return self.weight.dtype

class Tokenizer:
    special_tokens_map = {'pad_token': 'pad'}
    model_max_length = 128
    def get_added_vocab(self): return {}

def batch(prompt='lake', negative='', length=128):
    return SimpleNamespace(prompt=prompt, negative_prompt=negative, max_sequence_length=length,
                           extra={}, metrics=None, state=SimpleNamespace(extra={}))

class TextCacheTests(unittest.TestCase):
    def setUp(self):
        self.stage = text.EraserDiTEraseTextEncodingStage(Encoder(), Tokenizer())
        self.calls = 0
        def compute(b, args, **kwargs):
            self.calls += 1
            b.prompt_embeds = torch.full((1, 2, 3), float(len(b.prompt)))
            b.prompt_attention_mask = torch.ones(1, 2, dtype=torch.bool)
            b.negative_prompt_embeds = torch.full((1, 2, 3), float(len(b.negative_prompt)))
            b.negative_attention_mask = torch.ones(1, 2, dtype=torch.bool)
            b.state.extra['text_encoding'] = (kwargs['signature'], tuple(v.clone() for v in (
                b.prompt_embeds, b.prompt_attention_mask, b.negative_prompt_embeds, b.negative_attention_mask)))
            b.extra['text_encoding_cache_source'] = 'computed'
            return b
        self.stage._encode_uncached = compute
        self.state_patch = patch.object(text, 'get_task_state', lambda b: b.state)
        self.state_patch.start()
        self.env_patch = patch.dict(os.environ, {'MGERASE_TEXT_SESSION_CACHE': '1'})
        self.env_patch.start()
    def tearDown(self):
        self.env_patch.stop()
        self.state_patch.stop()
    def run_stage(self, b=None): return self.stage.forward(b or batch(), None)
    def test_hit_is_cpu_copy_and_close_clears(self):
        first, second = self.run_stage(), self.run_stage()
        self.assertEqual(self.calls, 1)
        self.assertTrue(second.extra['text_session_cache']['hit'])
        self.assertEqual(second.extra['text_encoding_cache_source'], 'session')
        self.assertTrue(torch.equal(first.prompt_embeds, second.prompt_embeds))
        second.prompt_embeds.zero_()
        third = self.run_stage()
        self.assertGreater(third.prompt_embeds.sum().item(), 0)
        self.assertTrue(all(v.device.type == 'cpu' for v in self.stage._session_text_cache[1]))
        self.stage.close()
        self.run_stage()
        self.assertEqual(self.calls, 2)
    def test_prompt_negative_length_and_model_updates_invalidate(self):
        self.run_stage()
        for b in (batch('bridge'), batch('bridge', 'blur'), batch('bridge', 'blur', 64)):
            self.run_stage(b)
        self.assertEqual(self.calls, 4)
        with torch.no_grad(): self.stage._text_encoder.weight.add_(1)
        self.run_stage(batch('bridge', 'blur', 64))
        self.assertEqual(self.calls, 5)
        self.stage._tokenizer.special_tokens_map['pad_token'] = 'new_pad'
        self.run_stage(batch('bridge', 'blur', 64))
        self.assertEqual(self.calls, 6)
    def test_disabled_or_training_never_reuses(self):
        with patch.dict(os.environ, {'MGERASE_TEXT_SESSION_CACHE': '0'}):
            self.run_stage(); self.run_stage()
        self.assertEqual(self.calls, 2)
        self.stage._text_encoder.train()
        self.run_stage(); self.run_stage()
        self.assertEqual(self.calls, 4)
        self.assertIsNone(self.stage._session_text_cache)
    def test_existing_window_cache_takes_priority(self):
        first = self.run_stage()
        b = batch()
        b.extra['cached_text_embeddings'] = {name: getattr(first, name).clone() for name in (
            'prompt_embeds', 'prompt_attention_mask', 'negative_prompt_embeds', 'negative_attention_mask')}
        self.run_stage(b)
        self.assertEqual(self.calls, 1)
        self.assertEqual(b.extra['text_encoding_cache_source'], 'window_runtime')
        self.assertFalse(b.extra['text_session_cache']['hit'])

class ArrayReader:
    def __init__(self, frames, **kwargs):
        self.frames = frames
        self.frames_read = 0
        self.channels = 1 if frames.ndim == 3 else 3
        self.read_into = kwargs.get('read_into', True)
        self.closed = False
    def read_frames(self, count):
        out = self.frames[self.frames_read:self.frames_read+count].copy()
        self.frames_read += count
        return out
    def read_frames_into(self, destination):
        count = destination.shape[0]
        destination[:] = self.frames[self.frames_read:self.frames_read+count]
        self.frames_read += count
    def close(self): self.closed = True

class MaskScanTests(unittest.TestCase):
    def run_case(self, frames):
        readers = []
        def factory(*args, **kwargs):
            reader = ArrayReader(frames, **kwargs); readers.append(reader); return reader
        with patch.object(video, 'SequentialVideoReader', factory):
            maximum, reader, stats = video.open_mask_reader_with_scan('mask', width=frames.shape[2], height=frames.shape[1], num_frames=len(frames), read_into=True, extract_red=frames.ndim==3)
        self.assertEqual(maximum, int((frames[..., 0] if frames.ndim==4 else frames).max()))
        self.assertEqual(reader.frames_read, 0)
        pieces=[]
        cursor=0
        for size in (5, 7, 9, 100):
            count=min(size, len(frames)-cursor)
            if count:
                pieces.append(reader.read_frames(count)); cursor += count
                self.assertEqual(reader.frames_read, cursor)
        self.assertTrue(np.array_equal(np.concatenate(pieces), frames))
        reader.close()
        self.assertTrue(all(r.closed for r in readers))
        return stats, readers
    def test_reuse_prefix_and_small_dark_clip(self):
        frames=np.arange(25*3*4,dtype=np.uint8).reshape(25,3,4)
        frames[2,0,0]=255
        stats, readers=self.run_case(frames)
        self.assertTrue(stats['reused']); self.assertEqual(len(readers),1)
        stats, readers=self.run_case(np.full((8,3,4),17,np.uint8))
        self.assertTrue(stats['reused']); self.assertEqual(stats['scanned_frames'],8)
    def test_dark_and_late_max_fall_back_bounded(self):
        for maximum in (150,255):
            frames=np.zeros((50,3,4),np.uint8); frames[42,0,0]=maximum
            stats, readers=self.run_case(frames)
            self.assertFalse(stats['reused']); self.assertEqual(len(readers),2)
            self.assertEqual(stats['retained_prefix_bytes'],0)
            self.assertEqual(stats['scanned_frames'],50 if maximum==150 else 48)
    def test_rgb_max_uses_red_only(self):
        frames=np.zeros((25,3,4,3),np.uint8); frames[...,1:]=255; frames[19,:,:,0]=42
        stats,_=self.run_case(frames)
        self.assertFalse(stats['reused']); self.assertEqual(stats['maximum'],42)
    def test_checkpoint_error_closes_reader(self):
        readers=[]
        def factory(*a,**k):
            r=ArrayReader(np.zeros((20,2,2),np.uint8)); readers.append(r); return r
        def cancelled(): raise RuntimeError('cancelled')
        with patch.object(video,'SequentialVideoReader',factory):
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                video.open_mask_reader_with_scan('mask',width=2,height=2,num_frames=20,checkpoint=cancelled)
        self.assertTrue(readers[0].closed)

if __name__=='__main__': unittest.main()
