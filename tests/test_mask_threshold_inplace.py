import unittest
from unittest import mock
import numpy as np
from utils.video_io import threshold_mask_array

def reference(a, threshold):
    return np.where(a <= threshold,np.uint8(0),np.uint8(255))

class ThresholdTests(unittest.TestCase):
    def test_all_byte_values_and_boundary_thresholds(self):
        original=np.arange(256,dtype=np.uint8).reshape(1,16,16)
        thresholds=[-float('inf'),-1.,0.,.3,38.,38.25,127.5,255.,256.,float('inf'),float('nan')]
        thresholds += [float(v) for v in range(256)]
        for t in thresholds:
            actual=original.copy()
            address=actual.ctypes.data
            result=threshold_mask_array(actual,t,inplace=True)
            np.testing.assert_array_equal(result,reference(original,t))
            self.assertIs(result,actual)
            self.assertEqual(actual.ctypes.data,address)
            self.assertEqual(actual.dtype,np.uint8)

    def test_out_of_place_preserves_input(self):
        source=np.arange(256,dtype=np.uint8).reshape(1,16,16)
        before=source.copy()
        result=threshold_mask_array(source,38.25)
        np.testing.assert_array_equal(source,before)
        np.testing.assert_array_equal(result,reference(source,38.25))
        self.assertFalse(np.shares_memory(source,result))

    def test_rgb_red_view_preserves_other_channels(self):
        rgb=np.random.default_rng(42).integers(0,256,(3,16,16,3),dtype=np.uint8)
        before=rgb.copy()
        red=rgb[...,0]
        result=threshold_mask_array(red,38.25,inplace=True)
        self.assertIs(result,red)
        np.testing.assert_array_equal(rgb[...,0],reference(before[...,0],38.25))
        np.testing.assert_array_equal(rgb[...,1:],before[...,1:])

    def test_negative_stride_and_fortran(self):
        source=np.random.default_rng(4).integers(0,256,(3,16,16),dtype=np.uint8)
        for a in (source.copy()[:,::-1,::-1],np.asfortranarray(source)):
            before=a.copy()
            result=threshold_mask_array(a,38.25,inplace=True)
            np.testing.assert_array_equal(result,reference(before,38.25))
            self.assertIs(result,a)

    def test_empty_mask(self):
        source=np.empty((0,16,16),np.uint8)
        self.assertIs(threshold_mask_array(source,38.25,inplace=True),source)

    def test_read_only_rejected_and_default_supported(self):
        source=np.arange(256,dtype=np.uint8).reshape(1,16,16)
        source.flags.writeable=False
        with self.assertRaises(ValueError):
            threshold_mask_array(source,38.25,inplace=True)
        np.testing.assert_array_equal(threshold_mask_array(source,38.25),reference(source,38.25))

    def test_non_uint8_inplace_rejected(self):
        source=np.arange(16,dtype=np.float32).reshape(1,4,4)
        before=source.copy()
        with self.assertRaises(ValueError):
            threshold_mask_array(source,3.5,inplace=True)
        np.testing.assert_array_equal(source,before)
        np.testing.assert_array_equal(threshold_mask_array(source,3.5),reference(source,3.5))

    def test_no_where_in_candidate(self):
        source=np.arange(256,dtype=np.uint8).reshape(1,16,16)
        expected=reference(source,38.25)
        with mock.patch('utils.video_io.np.where',side_effect=AssertionError('candidate allocated where output')):
            actual=threshold_mask_array(source,38.25,inplace=True)
        np.testing.assert_array_equal(actual,expected)

if __name__=='__main__':
    unittest.main()
