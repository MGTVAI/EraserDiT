import gc
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock
import weakref
sys.path.insert(0,str(Path.cwd()))
import numpy as np
from utils.video_io import ChunkedFrameCache, SequentialVideoWriter

class CacheOwnershipTests(unittest.TestCase):
    def cache(self,frames,start=17,sizes=(2,3,4)):
        cache=ChunkedFrameCache(start,frames.shape[1:],frames.dtype)
        cursor=0
        for count in sizes:
            cache.append_owned(frames[cursor:cursor+count].copy())
            cursor+=count
        return cache

    def test_complete_chunk_identity(self):
        frames=np.arange(5*2*3*3,dtype=np.uint8).reshape(5,2,3,3)
        cache=ChunkedFrameCache(17,frames.shape[1:],frames.dtype)
        cache.append_owned(frames)
        blocks,copied=cache.pop_before_owned(22)
        self.assertIs(blocks[0],frames)
        self.assertEqual(copied,0)
        self.assertEqual((cache.start_index,cache.end_index),(22,22))

    def test_multiblock_split_matches_original(self):
        frames=np.arange(9*2*3*3,dtype=np.uint8).reshape(9,2,3,3)
        old=self.cache(frames)
        new=self.cache(frames)
        expected=old.pop_before(25)
        blocks,copied=new.pop_before_owned(25)
        self.assertEqual([len(block) for block in blocks],[2,3,3])
        self.assertTrue(np.array_equal(np.concatenate(blocks),expected))
        self.assertEqual(copied,frames[8:].nbytes)
        self.assertEqual((new.start_index,new.end_index),(old.start_index,old.end_index))
        self.assertTrue(np.array_equal(new.slice(25,26),old.slice(25,26)))
        self.assertTrue(all(not np.shares_memory(block,new._chunks[0]) for block in blocks))

    def test_split_buffers_independent(self):
        frames=np.arange(5*2*3*3,dtype=np.uint8).reshape(5,2,3,3)
        cache=self.cache(frames,sizes=(5,))
        blocks,copied=cache.pop_before_owned(19)
        saved=blocks[0].copy()
        cache.overwrite(19,np.zeros_like(frames[2:]))
        self.assertTrue(np.array_equal(blocks[0],saved))
        blocks[0][...] = 255
        self.assertFalse(cache.slice(19,22).any())
        self.assertEqual(copied,frames[2:].nbytes)

    def test_repeated_boundary_empty_invalid(self):
        frames=np.arange(9*2*3*3,dtype=np.uint8).reshape(9,2,3,3)
        cache=self.cache(frames)
        self.assertEqual(cache.pop_before_owned(17),([],0))
        with self.assertRaises(ValueError):
            cache.pop_before_owned(27)
        arrays=[]
        for end in (19,22,26):
            blocks,copied=cache.pop_before_owned(end)
            arrays.extend(blocks)
            self.assertEqual(copied,0)
        self.assertTrue(np.array_equal(np.concatenate(arrays),frames))
        self.assertEqual(cache.pop_before_owned(26),([],0))

    def test_one_queue_task_and_independent_container(self):
        writer=SequentialVideoWriter.__new__(SequentialVideoWriter)
        writer.width=3
        writer.height=2
        writer._async_error=None
        writer._async_queue=object()
        writer._enqueue_async=Mock()
        blocks=[np.zeros((2,2,3,3),np.uint8),np.ones((3,2,3,3),np.uint8)]
        writer.write_frame_blocks_owned(blocks)
        writer._enqueue_async.assert_called_once()
        queued=writer._enqueue_async.call_args.args[0]
        self.assertIsNot(queued,blocks)
        self.assertIs(queued[0],blocks[0])
        blocks.clear()
        self.assertEqual(len(queued),2)

@unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg unavailable')
class WriterTests(unittest.TestCase):
    def writer(self,path,depth):
        return SequentialVideoWriter(path,width=16,height=16,fps=24.,codec_name='h264',thread_count=1,async_queue_depth=depth)

    def test_sync_async_block_stream_byte_exact(self):
        frames=np.random.default_rng(42).integers(0,256,(17,16,16,3),dtype=np.uint8)
        hashes=[]
        with tempfile.TemporaryDirectory() as directory:
            for depth in (0,1):
                for owned in (False,True):
                    path=Path(directory)/f'{depth}_{owned}.mp4'
                    writer=self.writer(path,depth)
                    if owned:
                        blocks=[frames[:4].copy(),np.asfortranarray(frames[4:11]),frames[11:].copy()]
                        writer.write_frame_blocks_owned(blocks)
                    else:
                        writer.write_frames_owned(frames.copy())
                    writer.close()
                    hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(len(set(hashes)),1)

    def test_release_after_batch_before_next_item(self):
        with tempfile.TemporaryDirectory() as directory:
            writer=self.writer(Path(directory)/'release.mp4',1)
            frames=np.zeros((4,16,16,3),np.uint8)
            reference=weakref.ref(frames)
            blocks=[frames]
            writer.write_frame_blocks_owned(blocks)
            blocks.clear()
            del frames
            writer._async_queue.join()
            gc.collect()
            self.assertIsNone(reference())
            writer.close()

    def test_async_failure_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            writer=self.writer(Path(directory)/'failure.mp4',1)
            writer._write_frames_sync=Mock(side_effect=RuntimeError('forced writer failure'))
            writer.write_frame_blocks_owned([np.zeros((4,16,16,3),np.uint8)])
            writer._async_queue.join()
            with self.assertRaisesRegex(RuntimeError,'asynchronous ffmpeg writer failed'):
                writer.close()

if __name__=='__main__':
    unittest.main(verbosity=2)
