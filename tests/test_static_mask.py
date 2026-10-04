"""Real ffmpeg checks for browser masks, including fractional FPS and alpha."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from PIL import Image

from entrypoints.server.static_mask import expand_uploaded_png
from entrypoints.server.task import ServiceError


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'requires ffmpeg')
class StaticMaskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root/'video.mp4'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',
                        'color=black:s=32x24:r=24000/1001','-frames:v','5',
                        '-c:v','libx264','-threads','1',str(self.video)], check=True)
        self.mask = self.root/'upload.mp4'

    def test_static_png_exact_frames_and_alpha(self):
        image = Image.new('RGBA',(32,24),(255,255,255,0))
        image.putpixel((8,9),(255,255,255,255))
        image.save(self.mask,format='PNG')
        output = expand_uploaded_png(self.video,self.mask)
        probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames',
            '-show_entries','stream=nb_read_frames,r_frame_rate,width,height','-of','json',str(output)]))['streams'][0]
        self.assertEqual(probe['nb_read_frames'],'5')
        self.assertEqual(probe['r_frame_rate'],'24000/1001')
        frames = subprocess.check_output(['ffmpeg','-v','error','-i',str(output),
            '-f','rawvideo','-pix_fmt','gray','-threads','1','pipe:1'])
        self.assertEqual(len(frames),32*24*5)
        for i in range(5):
            frame=frames[i*32*24:(i+1)*32*24]
            self.assertEqual(frame[9*32+8],255)
            self.assertEqual(sum(frame),255)

    def test_bad_dimensions_and_corrupt_png_rejected(self):
        Image.new('RGB',(16,16)).save(self.mask,format='PNG')
        with self.assertRaises(ServiceError):
            expand_uploaded_png(self.video,self.mask)
        self.mask.write_bytes(b'\x89PNG\r\n\x1a\ncorrupt')
        with self.assertRaises(ServiceError):
            expand_uploaded_png(self.video,self.mask)
        self.assertFalse((self.root/'mask_static.mkv').exists())

    def test_existing_video_is_untouched(self):
        self.mask.write_bytes(b'unchanged')
        self.assertEqual(expand_uploaded_png(self.video,self.mask),self.mask)
        self.assertEqual(self.mask.read_bytes(),b'unchanged')
