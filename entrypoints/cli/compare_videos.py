"""Stream RGB video comparisons with mask/edge and temporal error metrics."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


def probe(path):
    data = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames',
        '-of', 'json', str(path)], text=True))
    stream = data['streams'][0]
    return dict(width=int(stream['width']), height=int(stream['height']),
                frames=int(stream['nb_read_frames']), fps=stream['avg_frame_rate'])


def frame_metrics(reference, candidate, mask, previous_error=None):
    if reference.shape != candidate.shape or mask.shape != reference.shape[:2]:
        raise ValueError('reference, candidate and mask dimensions differ')
    a, b = reference.astype(np.float32), candidate.astype(np.float32)
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT_101)
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a*a)-ma*ma, blur(b*b)-mb*mb, blur(a*b)-ma*mb
    ssim = ((2*ma*mb+2.55**2)*(2*cov+7.65**2)) / ((ma*ma+mb*mb+2.55**2)*(va+vb+7.65**2))
    error = b-a
    kernel = np.ones((11, 11), dtype=np.uint8)
    edge = (cv2.dilate(mask.astype(np.uint8), kernel) != cv2.erode(mask.astype(np.uint8), kernel))
    result = dict(ssim=float(ssim.mean()), mae=float(np.abs(error).mean()), mse=float((error*error).mean()),
                  mask_pixels=int(mask.sum()), edge_pixels=int(edge.sum()))
    for name, region in (('mask', mask), ('edge', edge), ('outside', ~mask)):
        result[name+'_ssim'] = float(ssim[region].mean()) if region.any() else None
        result[name+'_mae'] = float(np.abs(error[region]).mean()) if region.any() else None
    result['temporal_delta_mae'] = float(np.abs(error-previous_error).mean()) if previous_error is not None else None
    return result, error


def aggregate(rows):
    if not rows:
        raise ValueError('cannot compare empty videos')
    keys = ('ssim', 'mae', 'mse', 'mask_ssim', 'edge_ssim', 'outside_ssim',
            'mask_mae', 'edge_mae', 'outside_mae', 'temporal_delta_mae')
    means = {}
    for key in keys:
        values = [r[key] for r in rows if r[key] is not None]
        means[key] = float(np.mean(values)) if values else None
    mask_rows = [r for r in rows if r['mask_ssim'] is not None]
    return dict(frames=len(rows), mask_frames=len(mask_rows), mean=means,
                min_frame_ssim=min(r['ssim'] for r in rows),
                min_mask_frame_ssim=min(r['mask_ssim'] for r in mask_rows) if mask_rows else None,
                worst_frame=min(range(len(rows)), key=lambda i: rows[i]['ssim']),
                worst_mask_frame=min((i for i, r in enumerate(rows) if r['mask_ssim'] is not None),
                                     key=lambda i: rows[i]['mask_ssim'], default=None))


def review_images(paths, frames, directory):
    directory.mkdir(parents=True, exist_ok=False)
    caps = [cv2.VideoCapture(str(p)) for p in paths]
    try:
        for index in sorted(set(frames)):
            images = []
            for cap in caps:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'cannot decode review frame {index}')
                images.append(frame)
            a, b, mask = images
            diff = np.clip(np.abs(b.astype(np.float32)-a)*8, 0, 255).astype(np.uint8)
            scale = min(1., 640/a.shape[1])
            size = (round(a.shape[1]*scale), round(a.shape[0]*scale))
            full = np.concatenate([cv2.resize(image, size) for image in (a, b, mask, diff)], axis=1)
            cv2.imwrite(str(directory / f'frame_{index:05d}.jpg'), full)
            ys, xs = np.where(mask.mean(axis=2) > 127)
            if len(xs):
                x1, x2 = max(0, xs.min()-32), min(a.shape[1], xs.max()+33)
                y1, y2 = max(0, ys.min()-32), min(a.shape[0], ys.max()+33)
                cv2.imwrite(str(directory / f'mask_{index:05d}.png'),
                            np.concatenate([im[y1:y2, x1:x2] for im in (a, b, mask, diff)], axis=1))
    finally:
        for cap in caps:
            cap.release()


def compare(reference, candidate, mask, *, review_dir=None):
    paths = [Path(p) for p in (reference, candidate, mask)]
    metadata = [probe(p) for p in paths]
    for meta in metadata[1:]:
        if any(meta[k] != metadata[0][k] for k in ('frames', 'width', 'height')):
            raise ValueError('video/mask frame counts or dimensions differ')
    if Fraction(metadata[0]['fps']) != Fraction(metadata[1]['fps']):
        raise ValueError('candidate frame rate differs from reference')
    # Mask frame-rate tags may be rounded; pair masks by exact frame index.
    caps = [cv2.VideoCapture(str(p)) for p in paths]
    hashes = [hashlib.sha256() for _ in paths]
    rows, previous = [], None
    try:
        for index in range(metadata[0]['frames']):
            images = []
            for cap, digest in zip(caps, hashes):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'early decode failure at frame {index}')
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                digest.update(frame.tobytes())
                images.append(frame)
            row, previous = frame_metrics(images[0], images[1], images[2].mean(axis=2) > 127, previous)
            rows.append(dict(index=index, **row))
        if any(cap.read()[0] for cap in caps):
            raise ValueError('decoded more frames than declared')
    finally:
        for cap in caps:
            cap.release()
    summary = aggregate(rows)
    if review_dir:
        indices = [0, len(rows)//2, len(rows)-1, summary['worst_frame']]
        if summary['worst_mask_frame'] is not None:
            indices.append(summary['worst_mask_frame'])
        # Include both sides of the standard 121-frame/9-overlap window seam.
        indices += [i for seam in range(112, len(rows), 112) for i in (seam-1, seam, seam+1) if i < len(rows)]
        review_images(paths, indices, Path(review_dir))
    return dict(**summary, per_frame=rows,
                decoded_rgb_sha256={key: digest.hexdigest() for key, digest in zip(('reference', 'candidate', 'mask'), hashes)},
                exact_rgb_match=hashes[0].digest() == hashes[1].digest(),
                paths={key: str(p.resolve()) for key, p in zip(('reference', 'candidate', 'mask'), paths)},
                metadata=metadata, protocol=dict(rgb_range=[0, 255], ssim_window=11, sigma=1.5,
                    border='reflect101', mask_threshold=127, edge='11x11 dilation minus erosion',
                    aggregation='mean of per-frame means; empty regions excluded',
                    temporal='mean absolute change of candidate-reference error; not motion compensated',
                    opencv=cv2.__version__, numpy=np.__version__), visual_status='not reviewed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--mask', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review-dir', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    cv2.setNumThreads(4)
    result = compare(args.reference, args.candidate, args.mask, review_dir=args.review_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'per_frame'}, indent=2))


if __name__ == '__main__':
    main()
