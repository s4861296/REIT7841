
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import av
import numpy as np
from PIL import Image
from r3d_native_video_store import sha256_file, validate_source_clock


SOURCE_KEYS = ('sample_id', 'frame_count', 'width', 'height', 'fps', 'fps_fraction',
               'mp4_path', 'mp4_sha256', 'frames')
FRAME_KEYS = ('frame_index', 'pts', 'time_base', 'source_time_s', 'png_path',
              'png_sha256', 'png_rgb_sha256', 'mp4_rgb_sha256')
IDENTITY_KEYS = {'sample_id', 'mode', 'native_shape', 'fps_fraction', 'rgb_sha256', 'source_time_sha256'}


def observation_descriptor(prepared):
    result = {key: prepared[key] for key in SOURCE_KEYS}
    result['frames'] = [{key: frame[key] for key in FRAME_KEYS} for frame in prepared['frames']]
    return result


def validate_descriptor(source):
    if set(source) != set(SOURCE_KEYS):
        raise ValueError('Observation-only source descriptor required')
    if any(type(source[k]) is not int or source[k] <= 0 for k in ('frame_count', 'width', 'height')):
        raise ValueError('Original positive native dimensions required')
    frames = source['frames']; n = source['frame_count']
    if len(frames) != n or any(set(f) != set(FRAME_KEYS) for f in frames):
        raise ValueError('Complete observation-only original frame inventory required')
    if any(type(f['frame_index']) is not int or f['frame_index'] != i for i, f in enumerate(frames)):
        raise ValueError('Missing or repeated source index')
    rate = Fraction(source['fps_fraction']); tick = Fraction(frames[0]['time_base'])
    if not np.isfinite(source['fps']) or abs(float(rate) - source['fps']) > 1e-4:
        raise ValueError('Original FPS differs from rational clock')
    if any(Fraction(f['time_base']) != tick for f in frames):
        raise ValueError('Changing source time base')
    validate_source_clock([f['pts'] for f in frames], tick, rate)
    times = np.asarray([float(Fraction(f['pts']) * tick) for f in frames], np.float64)
    if not np.array_equal(times, np.asarray([f['source_time_s'] for f in frames], np.float64)):
        raise ValueError('Original absolute source clock changed')
    return times


def load_encoding_observations(source, mode):
    times = validate_descriptor(source)
    if mode not in ('original_mp4', 'original_png'):
        raise ValueError('Explicit original input mode required')
    if sha256_file(source['mp4_path']) != source['mp4_sha256']:
        raise ValueError('Original encoded video mutated')
    n, h, w = (source[k] for k in ('frame_count', 'height', 'width'))
    rgb = np.empty((n, h, w, 3), np.uint8)
    if mode == 'original_png':
        for index, item in enumerate(source['frames']):
            path = Path(item['png_path'])
            before = sha256_file(path)
            if before != item['png_sha256']:
                raise ValueError('Prepared PNG file changed')
            with Image.open(path) as image:
                if image.mode != 'RGB' or image.size != (w, h):
                    raise ValueError('Native RGB PNG required; do not convert or resize')
                value = np.asarray(image)
                if value.dtype != np.uint8 or hashlib.sha256(value.tobytes()).hexdigest() != item['png_rgb_sha256']:
                    raise ValueError('Decoded PNG content differs')
                rgb[index] = value
            if sha256_file(path) != before:
                raise ValueError('PNG changed while decoding')
    else:
        decoded_count = 0
        with av.open(source['mp4_path']) as container:
            if len(container.streams.video) != 1:
                raise ValueError('Unambiguous original stream required')
            stream = container.streams.video[0]; stream.thread_count = 1
            if Fraction(stream.average_rate) != Fraction(source['fps_fraction']):
                raise ValueError('Source FPS changed')
            for index, frame in enumerate(container.decode(video=0)):
                if index >= n:
                    raise ValueError('Extra source frame')
                item = source['frames'][index]
                if (frame.pts != item['pts'] or Fraction(frame.time_base) != Fraction(item['time_base'])
                        or (frame.height, frame.width) != (h, w)):
                    raise ValueError('Original source coordinates changed')
                value = frame.to_ndarray(format='rgb24')
                if hashlib.sha256(value.tobytes()).hexdigest() != item['mp4_rgb_sha256']:
                    raise ValueError('Encoded RGB decode differs from preparation')
                rgb[index] = value; decoded_count += 1
        if decoded_count != n:
            raise ValueError('Missing source frames')
    if sha256_file(source['mp4_path']) != source['mp4_sha256']:
        raise ValueError('Source changed during decode')
    identity = dict(sample_id=source['sample_id'], mode=mode, native_shape=[n, h, w],
                    fps_fraction=source['fps_fraction'], rgb_sha256=hashlib.sha256(memoryview(rgb)).hexdigest(),
                    source_time_sha256=hashlib.sha256(times.tobytes()).hexdigest())
    return rgb, times, identity


def require_same_observation(native_identity, upstream_identity):
    if (set(native_identity) != IDENTITY_KEYS or set(upstream_identity) != IDENTITY_KEYS
            or native_identity != upstream_identity):
        raise ValueError('Mixed RGB/context source modes or coordinates; reject the comparison')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError('Preserve previous observation checks')
    code = {Path(__file__).name: Path(__file__).read_bytes(),
            'r3d_native_video_store.py': Path(__file__).with_name('r3d_native_video_store.py').read_bytes()}
    code_hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in code.items()}
    report_path = args.prepared / 'report.json'
    if sha256_file(report_path) != '13652f32b4bebba0e354804eadf49d0d1ab67bc4c4c24a524b74a243f671a12e':
        raise ValueError('Require the fixed complete Cache4 source preparation')
    prepared = json.loads(report_path.read_text())
    sources_path = args.prepared / 'sources.json'
    if sha256_file(sources_path) != prepared['sources_sha256']:
        raise ValueError('Prepared source list changed')
    sources = json.loads(sources_path.read_text())
    if len(sources) != 8:
        raise ValueError('Complete prepared population required')
    results = []
    for source in sources:
        descriptor = observation_descriptor(source)
        for mode in ('original_mp4', 'original_png'):
            rgb, times, identity = load_encoding_observations(descriptor, mode)
            require_same_observation(identity, identity.copy())
            results.append(identity)
            print(json.dumps(dict(source=source['sample_id'], mode=mode, frames=len(rgb))), flush=True)
            del rgb, times
    if 'torch' in sys.modules:
        raise RuntimeError('CPU loader must not import a model')
    for name, digest in code_hashes.items():
        if sha256_file(Path(__file__).with_name(name)) != digest:
            raise ValueError('Implementation changed during observation replay')
    if sha256_file(report_path) != '13652f32b4bebba0e354804eadf49d0d1ab67bc4c4c24a524b74a243f671a12e':
        raise ValueError('Preparation changed during observation replay')
    if sha256_file(sources_path) != prepared['sources_sha256']:
        raise ValueError('Prepared source list changed during observation replay')
    code_dir = args.output.parent / (args.output.stem + '_code')
    code_dir.mkdir(exist_ok=False)
    for name, raw in code.items():
        with (code_dir / name).open('xb') as stream:
            stream.write(raw)
    result = dict(completed=True, observation_replay_only=True, source_videos=8, input_mode_passes=16,
                  original_frame_uses=2880, identities=results, model_inference=False,
                  gpu_used=False, training_updates=0, gt_mask_files_opened=0,
                  paired_reference_in_forward=False, test_used=False, real_gt_used=False,
                  prepared_report_sha256=sha256_file(report_path), sources_sha256=sha256_file(sources_path),
                  code_sha256=code_hashes,
                  limitation='Source-mode plumbing check only. Upstream and native model stages not yet executed.')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
