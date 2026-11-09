
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
THRESHOLDS = (.03, .06, .12)


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
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def scope_helper():
    
    require(sha(SCOPE_HELPER) == SCOPE_HELPER_SHA256, 'Frozen native scope helper changed')
    spec = importlib.util.spec_from_file_location('_interpolation_native_scope', SCOPE_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_clock_receipt(row):
    require(sha(row['input_clock_path']) == row['input_clock_sha256'], 'Native clock SHA changed')
    clock = json.loads(Path(row['input_clock_path']).read_text())
    for key in ('source_frame_indices', 'source_pts', 'source_time_s'):
        require(np.array_equal(np.asarray(clock[key]), np.asarray(row[key])), 'Native clock mismatch: ' + key)
    base = Fraction(row['time_base_fraction'])
    expected = np.asarray([float(int(p) * base) for p in row['source_pts']], np.float64)
    require(np.array_equal(expected, np.asarray(row['source_time_s'], np.float64)), 'Exact PTS/time disagreement')


def residual_maps(left, actual, right, prediction, alpha):
    
    require(isinstance(prediction, np.ndarray) and prediction.dtype == np.float32
            and prediction.shape == actual.shape, 'Estimate must be same-shape float32 HWC RGB')
    require(np.isfinite(prediction).all() and (prediction >= 0).all() and (prediction <= 1).all(),
            'Estimate must be finite RGB in [0,1]')
    shape = actual.shape[:2]
    residual, before, after, linear = (np.zeros(shape, np.float32) for _ in range(4))
    scale = np.float32(255)
    a = np.float32(alpha)
    for channel in range(3):
        middle_channel = actual[:, :, channel].astype(np.float32) / scale
        left_channel = left[:, :, channel].astype(np.float32) / scale
        right_channel = right[:, :, channel].astype(np.float32) / scale
        residual += np.abs(prediction[:, :, channel] - middle_channel)
        before += np.abs(left_channel - middle_channel)
        after += np.abs(right_channel - middle_channel)
        linear += np.abs((np.float32(1) - a) * left_channel + a * right_channel - middle_channel)
    residual /= np.float32(3)
    before /= np.float32(3)
    after /= np.float32(3)
    linear /= np.float32(3)
    np.minimum(before, after, out=before)
    return residual, before, linear


def collect_frames(row, frames, estimate, progress=None):
    
    helper = scope_helper()
    x0, y0, x1, y1 = helper.validate_scope(row)
    require(callable(estimate), 'Explicit interpolation callback required')
    boxes, weights, grids = helper.tile_layout(row['height'], row['width'])
    n = row['frame_count']
    arrays = dict(residual_mean=np.full((n, 81), np.nan, np.float64),
        residual_fraction=np.full((n, 81, 3), np.nan, np.float64),
        raw_min_neighbor_mean=np.full((n, 81), np.nan, np.float64),
        linear_blend_residual_mean=np.full((n, 81), np.nan, np.float64),
        known=np.zeros(n, bool), alpha=np.full(n, np.nan, np.float64),
        residual_thresholds=np.asarray(THRESHOLDS, np.float64),
        tile_area_weights=weights, tile_grid=grids, tile_boxes_yxyx=np.asarray(boxes, np.int64),
        source_frame_indices=np.asarray(row['source_frame_indices'], np.int64),
        source_pts=np.asarray(row['source_pts'], np.int64), source_time_s=np.asarray(row['source_time_s'], np.float64),
        source_time_base=np.asarray(row['time_base_fraction']))
    base = Fraction(row['time_base_fraction'])
    window = []
    count = 0
    for pts, time_base, rgb in frames:
        require(count < n, 'Extra frame inside selected stream')
        require(type(pts) is int and pts == row['source_pts'][count] and Fraction(time_base) == base,
                'Missing, repeated, reordered or changed source PTS/timebase')
        require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8
                and rgb.shape == (row['source_height'], row['source_width'], 3), 'Original RGB geometry/dtype changed')
        window.append(rgb)
        count += 1
        if len(window) == 3:
            center = count - 2
            alpha = float(Fraction(row['source_pts'][center] - row['source_pts'][center - 1],
                                   row['source_pts'][center + 1] - row['source_pts'][center - 1]))
            require(0 < alpha < 1, 'Middle timestamp is not strictly inside its neighbors')
            left, middle, right = (image[y0:y1, x0:x1] for image in window)
            
            
            left_view, right_view = left.view(), right.view()
            left_view.setflags(write=False); right_view.setflags(write=False)
            prediction = estimate(left_view, right_view, alpha)
            residual, raw_min, linear = residual_maps(left, middle, right, prediction, alpha)
            for j, (top, start, bottom, end) in enumerate(boxes):
                region = residual[top:bottom, start:end]
                arrays['residual_mean'][center, j] = region.mean(dtype=np.float64)
                arrays['raw_min_neighbor_mean'][center, j] = raw_min[top:bottom, start:end].mean(dtype=np.float64)
                arrays['linear_blend_residual_mean'][center, j] = linear[top:bottom, start:end].mean(dtype=np.float64)
                for k, threshold in enumerate(THRESHOLDS):
                    arrays['residual_fraction'][center, j, k] = np.count_nonzero(region >= threshold) / region.size
            arrays['known'][center] = True
            arrays['alpha'][center] = alpha
            
            window.pop(0)
            del left, middle, right, left_view, right_view, prediction, residual, raw_min, linear, region
        if progress is not None and (count % 30 == 0 or count == n):
            progress(dict(stage='native_interpolation', decoded_frames=count, total_frames=n,
                          supported_frames=int(arrays['known'].sum())))
    require(count == n, 'Source ended before complete native scope')
    require(int(arrays['known'].sum()) == max(n - 2, 0), 'Unexpected supported center coverage')
    return arrays


def extract_native(row, estimate, progress=None):
    
    import av
    helper = scope_helper()
    helper.validate_scope(row)
    validate_clock_receipt(row)
    require(sha(row['video_path']) == row['video_sha256'], 'Source video SHA changed')
    began = time.monotonic()
    pts = row['source_pts']; base = Fraction(row['time_base_fraction'])
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
                require((frame.height, frame.width) == (row['source_height'], row['source_width']), 'Source resized')
                decoded += 1
                yield int(frame.pts), frame.time_base, frame.to_ndarray(format='rgb24')
                if decoded == len(pts):
                    break

    arrays = collect_frames(row, frames(), estimate, progress)
    receipt = dict(sample_id=row['sample_id'], source_video_sha256=row['video_sha256'],
        input_row_canonical_sha256=canonical_sha(row), input_clock_sha256=row['input_clock_sha256'],
        frame_count=decoded, supported_frame_count=int(arrays['known'].sum()), unknown_frame_count=int((~arrays['known']).sum()),
        source_dimensions=[row['source_height'], row['source_width']], input_dimensions=[row['height'], row['width']],
        crop_xyxy=row.get('crop_xyxy'), observer_sha256=sha(__file__),
        scope_helper_path=str(SCOPE_HELPER), scope_helper_sha256=SCOPE_HELPER_SHA256,
        source_pts_sha256=hashlib.sha256(arrays['source_pts'].tobytes()).hexdigest(),
        source_time_s_sha256=hashlib.sha256(arrays['source_time_s'].tobytes()).hexdigest(),
        pyav_version=av.__version__, numpy_version=np.__version__, elapsed_seconds=time.monotonic() - began,
        codec_preroll_discarded_frames=preroll, decode_threads=2, maximum_cached_source_RGB_frames=3,
        source_resize=False, FPS_resampling=False, selected_frames_dropped=False, RGB_scratch_written=False,
        labels_used=False, GT_ROI_used=False, model_training=False,
        callback_inputs='Only left/right original uint8 RGB viewport and alpha; middle frame never passed',
        callback_output='Same native HWC float32 RGB in [0,1]; finite; no implicit clipping',
        alpha='(middlePTS-leftPTS)/(rightPTS-leftPTS), original rational source clock; not fixed0.5',
        residual_definition='Per-pixel mean(abs(predicted_RGB-actual_RGB/255)) then native-region mean',
        raw_definition='Per-pixel min(mean(abs(left-actual))/255,mean(abs(right-actual))/255), then native-region mean',
        linear_control='Per-pixel RGB mean absolute residual of (1-alpha)*left/255 + alpha*right/255',
        fraction_thresholds=list(THRESHOLDS), threshold_comparison='>=',
        tile_layout='column0 global;1:17 4x4 row-major;17:81 8x8 row-major; exact integer native-pixel boundaries',
        area_weights='Native region pixel area / total viewport area; weights sum1 separately for global/4x4/8x8',
        unsupported_edges='First/last have known=False and NaN measurements, never replicated or classified Clean',
        GPU_policy='No model loaded here; caller controls callback device/weights/shared GPU lock',
        caveat='Interpolation error is observation evidence, not a bug label or proof of frame/pixel localization accuracy')
    return arrays, receipt
