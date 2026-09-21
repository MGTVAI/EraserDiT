#!/usr/bin/env python3
"""Compare decoded Y planes in cache smoke outputs, including erase/edge regions.

Example: python scripts/cache_compare.py --directory results/transformer_cache_smoke
--mask results/dynamic_offload_smoke/mask_33.mp4
Regions use EraserDiT's RGB mask threshold, with a 9x9 morphological boundary ring.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess

import cv2
import numpy as np


def decode_y(path):
    info = json.loads(subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height', '-of', 'json', str(path),
    ], check=True, capture_output=True, text=True).stdout)['streams'][0]
    w, h = info['width'], info['height']
    raw = subprocess.run([
        'ffmpeg', '-v', 'error', '-i', str(path), '-pix_fmt', 'yuv420p',
        '-vsync', '0', '-f', 'rawvideo', '-',
    ], check=True, capture_output=True).stdout
    stride = w * h * 3 // 2
    if len(raw) % stride:
        raise ValueError(f'incomplete decoded frame: {path}')
    return np.frombuffer(raw, np.uint8).reshape(-1, stride)[:, :w*h].reshape(-1, h, w).astype(np.float32)


def ssim_map(a, b):
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a*a)-ma*ma, blur(b*b)-mb*mb, blur(a*b)-ma*mb
    c1, c2 = (0.01*255)**2, (0.03*255)**2
    return ((2*ma*mb+c1)*(2*cov+c2))/((ma*ma+mb*mb+c1)*(va+vb+c2))


def decode_mask(path, shape, threshold):
    # Match EraserDiT's RGB first-channel decode and threshold relative to
    # its maximum. Luma >127 incorrectly excludes faint mask boundary pixels.
    raw = subprocess.run([
        'ffmpeg', '-v', 'error', '-i', str(path), '-pix_fmt', 'rgb24',
        '-vsync', '0', '-f', 'rawvideo', '-',
    ], check=True, capture_output=True).stdout
    info = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height', '-of', 'json', str(path),
    ]))['streams'][0]
    h, w = info['height'], info['width']
    if len(raw) % (h*w*3):
        raise ValueError('incomplete mask frame')
    values = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)[..., 0]
    if values.shape != shape:
        raise ValueError('mask and baseline must have exactly matching frame counts and geometry')
    return values > float(values.max()) * threshold / 2


def ffmpeg_whole(a, b):
    metrics = {}
    for name, channel in (('ssim', 'Y'), ('psnr', 'y')):
        proc = subprocess.run([
            'ffmpeg', '-hide_banner', '-i', str(a), '-i', str(b),
            '-lavfi', name, '-f', 'null', '-',
        ], check=True, capture_output=True, text=True)
        lines = [line for line in proc.stderr.splitlines() if 'Parsed_'+name in line]
        if not lines:
            raise ValueError(f'ffmpeg did not report {name}')
        value = float(re.search(r'\b'+channel+r':([0-9.inf]+)', lines[-1]).group(1))
        metrics['ssim_y' if name == 'ssim' else 'psnr_y_db'] = value if np.isfinite(value) else None
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--mask', type=Path, required=True)
    parser.add_argument('--mask-threshold', type=float, default=0.039)
    parser.add_argument('--baseline', default='off')
    parser.add_argument('--reference', type=Path, help='External reference video; overrides --baseline')
    parser.add_argument('--report-name', default='quality.json')
    parser.add_argument('--names', nargs='+', help='Output stems to compare; omit to scan the directory')
    args = parser.parse_args()
    paths = [args.directory / (name+'.mp4') for name in args.names] if args.names else sorted(args.directory.glob('*.mp4'))
    reference = args.reference or args.directory / (args.baseline+'.mp4')
    baseline = decode_y(reference)
    mask = decode_mask(args.mask, baseline.shape, args.mask_threshold)
    if mask.shape != baseline.shape:
        raise ValueError('mask and baseline must have exactly matching frame counts and geometry')
    kernel = np.ones((9, 9), np.uint8)
    edge = np.stack([cv2.dilate(m.astype(np.uint8), kernel) != cv2.erode(m.astype(np.uint8), kernel) for m in mask])
    regions = {'whole': np.ones_like(mask), 'masked': mask, 'unmasked': ~mask, 'edge_9x9': edge}
    report = {'metric': 'whole: ffmpeg Y SSIM/PSNR; regions: Gaussian SSIM 11x11 sigma1.5, global Y MSE',
              'shape': list(baseline.shape), 'baseline': str(reference.resolve()),
              'mask_threshold': args.mask_threshold, 'mask_decode': 'rgb24:first_channel', 'outputs': {}}
    for path in paths:
        y = decode_y(path)
        if y.shape != baseline.shape:
            raise ValueError(f'frame count or geometry mismatch: {path}')
        squared_error = (baseline-y)**2
        temporal_error = np.diff(baseline-y, axis=0)**2
        ssim = np.stack([ssim_map(a, b) for a, b in zip(baseline, y)])
        scores = {}
        per_frame = []
        for index, (error_frame, ssim_frame, mask_frame) in enumerate(zip(squared_error, ssim, mask)):
            mse = float(error_frame.mean())
            per_frame.append({'frame': index, 'ssim_y_gaussian': float(ssim_frame.mean()),
                              'psnr_y_db': float(10*np.log10(255**2/mse)) if mse else None,
                              'masked_ssim_y': float(ssim_frame[mask_frame].mean()) if mask_frame.any() else None})
        for name, region in regions.items():
            if not region.any():
                scores[name] = None
                continue
            mse = float(squared_error[region].mean())
            scores[name] = {'psnr_y_db': float(10*np.log10(255**2/mse)) if mse else None,
                            'ssim_y': float(ssim[region].mean()), 'identical': mse == 0}
            temporal_region = region[1:] & region[:-1]
            scores[name]['temporal_delta_rmse_y'] = (
                float(np.sqrt(temporal_error[temporal_region].mean())) if temporal_region.any() else None
            )
        scores['whole']['ssim_y_gaussian'] = scores['whole']['ssim_y']
        scores['whole'].update(ffmpeg_whole(reference, path))
        report['outputs'][path.stem] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                      'regions': scores, 'per_frame': per_frame}
    target = args.directory / args.report_name
    target.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
