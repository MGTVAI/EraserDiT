#!/usr/bin/env python3
"""Streaming RGB video quality gate; lossy optimizations require visual review."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def frame_metrics(reference, candidate):
    if reference.shape != candidate.shape or reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError('matching three-channel frames required')
    a, b = reference.astype(np.float32), candidate.astype(np.float32)
    error = a - b
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a*a)-ma*ma, blur(b*b)-mb*mb, blur(a*b)-ma*mb
    c1, c2 = 2.55**2, 7.65**2
    ssim = ((2*ma*mb+c1)*(2*cov+c2))/((ma*ma+mb*mb+c1)*(va+vb+c2))
    return dict(ssim=float(ssim.mean(dtype=np.float64)),
                mse=float(np.square(error).mean(dtype=np.float64)),
                mae=float(np.abs(error).mean(dtype=np.float64)))


def passes(metrics):
    return metrics['ssim'] >= .985 and metrics['mse'] <= 36 and metrics['mae'] <= 6


def compare(reference, candidate, lossy=False):
    sources = [cv2.VideoCapture(str(path)) for path in (reference, candidate)]
    frames = []
    shape = None
    try:
        if not all(s.isOpened() for s in sources):
            raise ValueError('cannot open input video')
        for source in sources:
            source.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        fps = [s.get(cv2.CAP_PROP_FPS) for s in sources]
        if abs(fps[0]-fps[1]) > .001:
            raise ValueError('video frame rates differ')
        expected = [int(s.get(cv2.CAP_PROP_FRAME_COUNT)) for s in sources]
        if expected[0] != expected[1]:
            raise ValueError('video frame counts differ')
        while True:
            (ok_a, a), (ok_b, b) = [s.read() for s in sources]
            if ok_a != ok_b:
                raise ValueError('decoded video frame counts differ')
            if not ok_a:
                break
            if shape is None:
                shape = list(a.shape)
            if list(a.shape) != shape or list(b.shape) != shape:
                raise ValueError('frame geometry differs')
            # OpenCV BGR ordering does not change the channel-averaged metrics.
            metrics = frame_metrics(a, b)
            frames.append(dict(frame=len(frames), **metrics, threshold_pass=passes(metrics)))
        if not frames or len(frames) != expected[0]:
            raise ValueError('empty or incompletely decoded video')
    finally:
        for source in sources:
            source.release()
    whole = {k:float(np.mean([f[k] for f in frames])) for k in ('ssim','mse','mae')}
    return dict(reference=str(Path(reference).resolve()), candidate=str(Path(candidate).resolve()),
                metric='RGB channels, 0..255; Gaussian SSIM 11x11 sigma1.5 reflect border; equal frame/pixel mean',
                thresholds=dict(ssim_min=.985, mse_max=36, mae_max=6),
                shape=shape, frame_count=len(frames), whole=whole,
                threshold_pass=passes(whole), accepted=None if lossy else passes(whole),
                verdict='visual_review_required' if lossy else ('pass' if passes(whole) else 'fail'),
                optimization_class='lossy_cache_or_quantization' if lossy else 'non_cache_non_quantization',
                worst_frame_ssim=min(f['ssim'] for f in frames), per_frame=frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--lossy', action='store_true')
    args = parser.parse_args()
    cv2.setNumThreads(2)
    result = compare(args.reference, args.candidate, args.lossy)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('whole','threshold_pass','accepted','verdict')}))
    if result['accepted'] is False:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
