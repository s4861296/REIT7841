
from __future__ import annotations

from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


SCOPE_HELPER = (Path(__file__).resolve().parents[2] /
                'real_calibration_20261014_v1/photometric_v1/native_photometric.py')
SCOPE_HELPER_SHA256 = '5df50648f8c68adc51c2e4a0f2035f6d91dadbffddbe9883a99192ef98b717c9'
SCHEMA_VERSION = 'native_rgb_histogram_transport_v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def scope_helper():
    
    require(sha(SCOPE_HELPER) == SCOPE_HELPER_SHA256, 'Frozen native scope helper changed')
    spec = importlib.util.spec_from_file_location('_histogram_transport_scope', SCOPE_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _exact_times(row):
    base = Fraction(row['time_base_fraction'])
    expected = np.asarray([float(int(p) * base) for p in row['source_pts']], np.float64)
    require(np.array_equal(expected, np.asarray(row['source_time_s'], np.float64)),
            'Exact PTS/time disagreement')


def validate_clock_receipt(row):
    require(sha(row['input_clock_path']) == row['input_clock_sha256'], 'Native clock SHA changed')
    clock = json.loads(Path(row['input_clock_path']).read_text())
    for key in ('source_frame_indices', 'source_pts', 'source_time_s'):
        require(np.array_equal(np.asarray(clock[key]), np.asarray(row[key])),
                'Native clock mismatch: ' + key)
    _exact_times(row)


def _rgb(rgb, shape=None):
    require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3
            and rgb.shape[2] == 3 and min(rgb.shape[:2]) >= 8, 'Native uint8 HWC RGB required')
    if shape is not None:
        require(rgb.shape == shape, 'Original RGB geometry changed')


def _aggregate_fine(fine):
    
    tail = fine.shape[1:]
    coarse = fine.reshape((4, 2, 4, 2) + tail).sum(axis=(1, 3), dtype=np.int64)
    return np.concatenate((fine.sum(axis=0, dtype=np.int64)[None],
                           coarse.reshape((16,) + tail), fine), axis=0)


def _histograms(rgb, boxes):
    
    fine = np.empty((64, 3, 256), np.int64)
    for j, (y0, x0, y1, x1) in enumerate(boxes[17:81]):
        for channel in range(3):
            fine[j, channel] = np.bincount(rgb[y0:y1, x0:x1, channel].ravel(), minlength=256)
    return _aggregate_fine(fine)


def _pair_from_histograms(left, right, left_counts, right_counts, boxes):
    area = np.asarray([(y1-y0)*(x1-x0) for y0, x0, y1, x1 in boxes], np.int64)
    
    cdf_delta = np.cumsum(left_counts - right_counts, axis=2, dtype=np.int64)
    h_numerator = np.abs(cdf_delta[:, :, :-1]).sum(axis=(1, 2), dtype=np.int64)
    raw_fine = np.zeros(64, np.int64)
    for channel in range(3):
        
        delta = np.subtract(left[:, :, channel], right[:, :, channel], dtype=np.int16)
        np.abs(delta, out=delta)
        for j, (y0, x0, y1, x1) in enumerate(boxes[17:81]):
            raw_fine[j] += delta[y0:y1, x0:x1].sum(dtype=np.int64)
        del delta
    denominator = area.astype(np.float64) * (3 * 255)
    h = h_numerator / denominator
    d = _aggregate_fine(raw_fine) / denominator
    require(np.isfinite(h).all() and np.isfinite(d).all()
            and (h >= 0).all() and (d <= 1).all() and (h <= d + 1e-14).all(),
            'Histogram transport/raw RGB L1 invariant failed')
    return h, d


def pair_statistics(left, right):
    
    _rgb(left)
    _rgb(right, left.shape)
    boxes, _, _ = scope_helper().tile_layout(*left.shape[:2])
    return _pair_from_histograms(left, right, _histograms(left, boxes),
                                 _histograms(right, boxes), boxes)


def collect_frames(row, frames, progress=None):
    
    helper = scope_helper()
    x0, y0, x1, y1 = helper.validate_scope(row)
    _exact_times(row)
    boxes, weights, grids = helper.tile_layout(row['height'], row['width'])
    n = row['frame_count']
    arrays = dict(H_wasserstein=np.full((n, 2, 81), np.nan, np.float64),
        D_rgb_l1=np.full((n, 2, 81), np.nan, np.float64), known=np.zeros((n, 2), bool),
        center_known=np.zeros(n, bool), pair_time_delta_s=np.full((n, 2), np.nan, np.float64),
        neighbor_source_frame_indices=np.full((n, 2), -1, np.int64),
        tile_area_weights=weights, tile_grid=grids, tile_boxes_yxyx=np.asarray(boxes, np.int64),
        source_frame_indices=np.asarray(row['source_frame_indices'], np.int64),
        source_pts=np.asarray(row['source_pts'], np.int64),
        source_time_s=np.asarray(row['source_time_s'], np.float64),
        source_time_base=np.asarray(row['time_base_fraction']))
    base = Fraction(row['time_base_fraction'])
    previous = previous_counts = None
    count = 0
    for pts, time_base, rgb in frames:
        require(count < n, 'Extra frame inside selected stream')
        require(type(pts) is int and pts == row['source_pts'][count] and Fraction(time_base) == base,
                'Missing, repeated, reordered or changed source PTS/timebase')
        _rgb(rgb, (row['source_height'], row['source_width'], 3))
        current = rgb[y0:y1, x0:x1]
        counts = _histograms(current, boxes)
        if previous is not None:
            h, d = _pair_from_histograms(previous, current, previous_counts, counts, boxes)
            arrays['H_wasserstein'][count, 0] = arrays['H_wasserstein'][count-1, 1] = h
            arrays['D_rgb_l1'][count, 0] = arrays['D_rgb_l1'][count-1, 1] = d
            arrays['known'][count, 0] = arrays['known'][count-1, 1] = True
            elapsed = float((pts - row['source_pts'][count-1]) * base)
            arrays['pair_time_delta_s'][count, 0] = arrays['pair_time_delta_s'][count-1, 1] = elapsed
            arrays['neighbor_source_frame_indices'][count, 0] = row['source_frame_indices'][count-1]
            arrays['neighbor_source_frame_indices'][count-1, 1] = row['source_frame_indices'][count]
        previous, previous_counts = current, counts
        count += 1
        if progress is not None and (count % 30 == 0 or count == n):
            progress(dict(stage='native_histogram_transport', decoded_frames=count, total_frames=n,
                          observed_adjacent_pairs=max(count-1, 0)))
        
        del current, rgb, counts
    require(count == n, 'Source ended before complete native scope')
    arrays['center_known'][:] = arrays['known'].all(axis=1)
    require(int(arrays['known'].sum()) == 2 * max(n-1, 0), 'Incomplete adjacent-pair coverage')
    require(int(arrays['center_known'].sum()) == max(n-2, 0), 'Incomplete two-sided coverage')
    return arrays


def extract_native(row, progress=None):
    
    helper = scope_helper()
    helper.validate_scope(row)
    validate_clock_receipt(row)
    require(sha(row['video_path']) == row['video_sha256'], 'Source video SHA changed')
    import av
    began = time.monotonic()
    pts = row['source_pts']
    base = Fraction(row['time_base_fraction'])
    preroll, decoded = 0, 0

    def frames():
        nonlocal preroll, decoded
        with av.open(row['video_path']) as container:
            require(len(container.streams.video) == 1, 'Exactly one source video stream required')
            stream = container.streams.video[0]
            stream.thread_count = 2
            require(Fraction(stream.time_base) == base, 'Original source timebase changed')
            if pts[0] > 0:
                container.seek(pts[0], stream=stream, backward=True, any_frame=False)
            for frame in container.decode(video=0):
                require(frame.pts is not None and Fraction(frame.time_base) == base, 'Missing source clock')
                if frame.pts < pts[0]:
                    preroll += 1
                    continue
                if frame.pts > pts[-1]:
                    break
                require((frame.height, frame.width) == (row['source_height'], row['source_width']),
                        'Source resized')
                decoded += 1
                yield int(frame.pts), frame.time_base, frame.to_ndarray(format='rgb24')
                if decoded == len(pts):
                    break

    arrays = collect_frames(row, frames(), progress)
    receipt = dict(schema_version=SCHEMA_VERSION, sample_id=row['sample_id'],
        source_video_sha256=row['video_sha256'], input_row_canonical_sha256=canonical_sha(row),
        input_clock_sha256=row['input_clock_sha256'], frame_count=decoded,
        adjacent_pair_count=max(decoded-1, 0), supported_side_count=int(arrays['known'].sum()),
        supported_center_count=int(arrays['center_known'].sum()),
        unknown_center_count=int((~arrays['center_known']).sum()),
        source_dimensions=[row['source_height'], row['source_width']],
        input_dimensions=[row['height'], row['width']], crop_xyxy=row.get('crop_xyxy'),
        observer_sha256=sha(__file__), scope_helper_path=str(SCOPE_HELPER),
        scope_helper_sha256=SCOPE_HELPER_SHA256,
        source_pts_sha256=hashlib.sha256(arrays['source_pts'].tobytes()).hexdigest(),
        source_time_s_sha256=hashlib.sha256(arrays['source_time_s'].tobytes()).hexdigest(),
        pyav_version=av.__version__, numpy_version=np.__version__, elapsed_seconds=time.monotonic()-began,
        codec_preroll_discarded_frames=preroll, decode_threads=2, maximum_cached_source_RGB_frames=2,
        source_resize=False, FPS_resampling=False, selected_frames_dropped=False, RGB_scratch_written=False,
        labels_used=False, GT_ROI_used=False, model_training=False, GPU=False,
        histogram_bins=256, color_space='Original gamma-coded decoded RGB24, separate channels; no Y/luma conversion',
        H_definition='sum(channel,bin0..254,abs(cumulative_count_a-cumulative_count_b))/(3*255*region_pixel_count)',
        D_definition='sum(pixel,channel,abs(int16_RGB_a-int16_RGB_b))/(3*255*region_pixel_count)',
        units='Both normalized RGB channel-mean L1 in [0,1]; not L-infinity; not divided by elapsed time',
        invariant='0 <= H <= D <= 1; identical color distributions may give H=0 despite D>0',
        side_order=['previous_source_frame', 'next_source_frame'],
        tile_layout='column0 global;1:17 4x4 row-major;17:81 8x8 row-major; integer native-pixel boundaries',
        area_weights='Native region pixel area / total viewport area; each of global/4x4/8x8 sums to1',
        unknown_policy='Absent neighbor gives known=False, H/D/time_delta=NaN and neighbor_index=-1; center_known requires both sides',
        scope='Every original selected frame and exact PTS retained; only approved H13 viewport can crop',
        caveat='Histogram similarity alone cannot veto defects: spatial rearrangement or color-channel dependence may be abnormal')
    return arrays, receipt
