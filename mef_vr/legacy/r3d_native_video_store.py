
import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import mmap
from pathlib import Path
import shutil

import av
import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_clock(pts, time_base, fps):
    
    if not pts or any(type(p) is not int for p in pts):
        raise ValueError('Every original frame needs an integer source PTS')
    rate, tick = Fraction(fps), Fraction(time_base)
    if rate <= 0 or tick <= 0:
        raise ValueError('Positive original FPS and time base required')
    period = 1 / rate
    tolerance = min(tick, period / 100)
    for index, point in enumerate(pts):
        if index and point <= pts[index - 1]:
            raise ValueError('Repeated/non-monotonic original PTS')
        if abs((point - pts[0]) * tick - index * period) > tolerance:
            raise ValueError('Source is not CFR-compatible; no resampling allowed')
        if index and abs((point - pts[index - 1]) * tick - period) > tolerance:
            raise ValueError('Source interval differs from original FPS')


def scan_source(path):
    points, times, shape = [], [], None
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError('Select an unambiguous single-video-stream source')
        stream = container.streams.video[0]
        stream.thread_count = 2
        if stream.average_rate is None or stream.time_base is None:
            raise ValueError('Missing original FPS or time base')
        rate, tick = Fraction(stream.average_rate), Fraction(stream.time_base)
        metadata_count = stream.frames
        for frame in container.decode(video=0):
            if frame.pts is None or frame.time is None or Fraction(frame.time_base) != tick:
                raise ValueError('Missing or changing original timestamp basis')
            current = (frame.height, frame.width, 3)
            if shape is None:
                shape = current
            if shape != current or min(current) < 1:
                raise ValueError('Changing or invalid original dimensions')
            points.append(int(frame.pts))
            times.append(float(frame.time))
    validate_source_clock(points, tick, rate)
    return dict(frame_count=len(points), height=shape[0], width=shape[1],
                fps=float(rate), fps_fraction=str(rate), time_base_fraction=str(tick),
                source_pts=points, source_time_s=times,
                container_frame_count_not_used_for_allocation=metadata_count)


def _save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def create_native_video_store(source, output, *, expected_sha256=None,
                              reserve_bytes=40 * 1024**3, flush_frames=16):
    
    source, output = Path(source).resolve(strict=True), Path(output).absolute()
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError('Invalid disk reserve')
    if type(flush_frames) is not int or flush_frames < 1:
        raise ValueError('Invalid bounded flush interval')
    if not source.is_file() or output.exists() or not output.parent.is_dir():
        raise ValueError('Require a source file and new store under an existing parent')
    before = sha256_file(source)
    if expected_sha256 is not None and before != expected_sha256:
        raise ValueError('Original video hash differs')
    meta = scan_source(source)
    n, h, w = meta['frame_count'], meta['height'], meta['width']
    
    needed = n * h * w * 3 + n * 128 + 1024 * 1024
    if shutil.disk_usage(output.parent).free < needed + reserve_bytes:
        raise OSError('Insufficient disk space above reserved free bytes')
    if sha256_file(source) != before:
        raise ValueError('Source mutated during first decode')
    output.mkdir(exist_ok=False)
    rgb = None
    try:
        _save_json(output / 'status.json', dict(status='decoding_native_frames',
                   source=str(source), frame_count=n, reserve_bytes=reserve_bytes))
        rgb = np.lib.format.open_memmap(output / 'rgb.npy', mode='w+',
                                       dtype=np.uint8, shape=(n, h, w, 3))
        seen, decoded_hash = 0, hashlib.sha256()
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            stream.thread_count = 2
            if (str(Fraction(stream.average_rate)) != meta['fps_fraction'] or
                    str(Fraction(stream.time_base)) != meta['time_base_fraction']):
                raise ValueError('Source metadata changed on second decode')
            for frame in container.decode(video=0):
                if (seen >= n or frame.pts != meta['source_pts'][seen] or
                        frame.time is None or float(frame.time) != meta['source_time_s'][seen]):
                    raise ValueError('Source frame coverage/PTS changed on second decode')
                pixels = frame.to_ndarray(format='rgb24')
                if pixels.shape != (h, w, 3) or pixels.dtype != np.uint8:
                    raise ValueError('Original RGB dimensions changed')
                pixels = np.ascontiguousarray(pixels)
                decoded_hash.update(memoryview(pixels))
                rgb[seen] = pixels
                seen += 1
                if seen % flush_frames == 0:
                    rgb.flush()
                    if hasattr(rgb._mmap, 'madvise') and hasattr(mmap, 'MADV_DONTNEED'):
                        rgb._mmap.madvise(mmap.MADV_DONTNEED)
                    if shutil.disk_usage(output).free < reserve_bytes:
                        raise OSError('Disk reserve crossed during decode')
            if seen != n:
                raise ValueError('Incomplete second decode')
        rgb.flush()
        rgb._mmap.close()
        rgb = None
        if sha256_file(source) != before:
            raise ValueError('Original video mutated during decode')
        np.save(output / 'source_time_s.npy', np.asarray(meta['source_time_s'], np.float64),
                allow_pickle=False)
        meta.update(protocol='r3d_native_rgb_store_v1', completed=True, source=str(source),
                    source_sha256=before, rgb_dtype='uint8', rgb_format='PyAV rgb24',
                    decoded_rgb_sha256=decoded_hash.hexdigest(), decode_passes=2,
                    all_source_frames=True, resized=False, resampled=False,
                    labels_or_paired_clean_read=False, model_inference=False,
                    disk_reserve_bytes=reserve_bytes, flush_frames=flush_frames,
                    input_module_sha256=sha256_file(__file__),
                    artifacts={name: sha256_file(output / name)
                               for name in ('rgb.npy', 'source_time_s.npy')})
        _save_json(output / 'manifest.json', meta)
        _save_json(output / 'status.json', dict(status='completed', frame_count=n,
                   source_sha256=before, rgb_bytes=n*h*w*3))
        return meta
    except BaseException as error:
        if rgb is not None:
            rgb._mmap.close()
        _save_json(output / 'status.json', dict(status='failed', error=repr(error)))
        raise


@dataclass
class NativeVideoStore:
    rgb: np.memmap
    source_time_s: np.ndarray
    manifest: dict

    def close(self):
        self.rgb._mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def open_native_video_store(output):
    
    output = Path(output)
    meta = json.loads((output / 'manifest.json').read_text())
    status = json.loads((output / 'status.json').read_text())
    if (meta.get('protocol') != 'r3d_native_rgb_store_v1' or
            meta.get('completed') is not True or status.get('status') != 'completed'):
        raise ValueError('Require completed native store')
    if set(meta['artifacts']) != {'rgb.npy', 'source_time_s.npy'}:
        raise ValueError('Unexpected native artifacts')
    for name, digest in meta['artifacts'].items():
        if sha256_file(output / name) != digest:
            raise ValueError('Stored artifact hash mismatch: ' + name)
    validate_source_clock(meta['source_pts'], meta['time_base_fraction'], meta['fps_fraction'])
    times = np.load(output / 'source_time_s.npy', allow_pickle=False)
    shape = (meta['frame_count'], meta['height'], meta['width'], 3)
    if (times.dtype != np.float64 or times.shape != (shape[0],) or
            not np.array_equal(times, meta['source_time_s']) or len(meta['source_pts']) != shape[0]):
        raise ValueError('Stored clock differs from manifest')
    rgb = np.load(output / 'rgb.npy', mmap_mode='c', allow_pickle=False)
    if rgb.dtype != np.uint8 or rgb.shape != shape:
        rgb._mmap.close()
        raise ValueError('Stored native RGB differs from manifest')
    return NativeVideoStore(rgb, times, meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-sha256')
    parser.add_argument('--reserve-gib', type=int, default=40)
    args = parser.parse_args()
    result = create_native_video_store(args.video, args.output,
                expected_sha256=args.expected_sha256, reserve_bytes=args.reserve_gib * 1024**3)
    print(json.dumps(dict(store=str(args.output), frames=result['frame_count'],
                         height=result['height'], width=result['width'], fps=result['fps'])))


if __name__ == '__main__':
    main()
