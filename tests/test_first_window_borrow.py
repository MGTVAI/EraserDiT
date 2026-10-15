import unittest
import numpy as np
from utils.video_io import ChunkedFrameCache
from pipelines.runtime.windowing.commit_ops import splice_uint8_patch


class HeadOwnershipTests(unittest.TestCase):
    def cache(self):
        a = np.arange(8*2*3*3, dtype=np.uint8).reshape(8,2,3,3).copy()
        c = ChunkedFrameCache(0,(2,3,3),np.uint8); c.append_owned(a)
        return c,a
    def test_borrow_has_no_copy_or_cache_change(self):
        c,a = self.cache()
        self.assertIs(c.borrow_head(0,8),a)
        self.assertEqual((c.start_index,c.end_index),(0,8))
        self.assertIsNone(c.borrow_head(1,8)); self.assertIsNone(c.borrow_head(0,7))
    def test_tail_is_independent_before_inplace_patch(self):
        c,a = self.cache(); original=a.copy()
        c.append_owned(np.full((2,2,3,3),251,dtype=np.uint8))
        self.assertEqual(c.detach_borrowed_head(a,6),2*2*3*3)
        self.assertEqual((c.start_index,c.end_index),(6,10))
        patch_frames=np.full((8,2,1,3),200,dtype=np.uint8)
        modified,n=splice_uint8_patch(a,patch_frames,(1,0,1,2),source_is_exclusive=True)
        expected=original.copy();expected[:,:,1:2]=200
        np.testing.assert_array_equal(modified,expected)
        np.testing.assert_array_equal(c.slice(6,8),original[6:])
        self.assertEqual(n,0)
    def test_full_transfer_removes_head(self):
        c,a=self.cache();self.assertEqual(c.detach_borrowed_head(a,8),0)
        self.assertEqual((c.start_index,c.end_index),(8,8))
    def test_readonly_and_noncontiguous_use_fallback(self):
        c,a=self.cache();a.flags.writeable=False
        self.assertIsNone(c.borrow_head(0,8))
        c._chunks[0]=a[:, :, ::-1];self.assertIsNone(c.borrow_head(0,8))
    def test_wrong_owner_or_range_leaves_cache_intact(self):
        c,a=self.cache()
        for value,frontier in [(a.copy(),6),(a,9)]:
            with self.assertRaises(ValueError):c.detach_borrowed_head(value,frontier)
            self.assertIs(c._chunks[0],a);self.assertEqual(c.start_index,0)


if __name__=='__main__':unittest.main()
