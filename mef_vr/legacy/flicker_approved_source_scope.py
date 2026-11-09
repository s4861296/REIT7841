
from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from flicker_multimodel_streaming import approved_frame_interval


VERSION = 'approved_original_source_scope_v1'
CLOCK_KEYS = {'frame_count', 'height', 'width', 'fps', 'fps_fraction',
              'time_base_fraction', 'source_pts', 'source_time_s'}
CLOCK_OPTIONAL = {'cfr_compatible', 'clock_policy'}
FALSE_FLAGS = {'spatial_resize', 'temporal_sampling', 'fps_resampled',
               'video_reencoded', 'labels_loaded', 'paired_clean_used'}
SOURCE_KEYS = {'sample_id', 'video_path', 'video_sha256', 'approved_seconds'}
OBS_KEYS = CLOCK_KEYS | FALSE_FLAGS | {'video_path', 'video_sha256', 'rgb_sha256',
    'source_frame_count', 'source_start_frame', 'source_end_frame_exclusive', 'approved_seconds'}
OBS_OPTIONAL = CLOCK_OPTIONAL | {'sample_id', 'rgb_npy_sha256', 'vfr_caveat'}
SCOPE_KEYS = {'version', 'kind', 'approved_seconds', 'source_frame_count',
    'source_start_frame', 'source_end_frame_exclusive', 'input_frame_count',
    'selection_rule', 'input_context_policy', 'input_clock_sha256'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _seconds(value):
    require(isinstance(value, (list, tuple)) and len(value) == 2
            and all(type(v) in (int, float) and math.isfinite(v) for v in value)
            and 0 <= value[0] < value[1], 'Explicit finite increasing approved seconds required')
    return [float(v) for v in value]


def _clock(clock):
    require(type(clock) is dict and CLOCK_KEYS <= set(clock) <= CLOCK_KEYS | CLOCK_OPTIONAL,
            'Source-only clock whitelist required')
    require(all(type(clock[k]) is int and clock[k] > 0 for k in ('frame_count', 'height', 'width')),
            'Positive original integer geometry required')
    require(type(clock['fps']) in (int, float) and math.isfinite(clock['fps'])
            and isinstance(clock['fps_fraction'], str) and isinstance(clock['time_base_fraction'], str),
            'Original rational FPS/time base required')
    try:
        rate, tick = Fraction(clock['fps_fraction']), Fraction(clock['time_base_fraction'])
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError('Invalid original rational clock') from error
    require(rate > 0 and tick > 0 and float(rate) == clock['fps'], 'Original FPS differs')
    n, pts, times = clock['frame_count'], clock['source_pts'], clock['source_time_s']
    require(isinstance(pts, list) and isinstance(times, list) and len(pts) == len(times) == n
            and all(type(p) is int for p in pts) and all(b > a for a, b in zip(pts, pts[1:]))
            and all(type(t) in (int, float) and math.isfinite(t) for t in times)
            and times == [float(p * tick) for p in pts], 'Complete absolute original PTS/time required')
    require('cfr_compatible' not in clock or type(clock['cfr_compatible']) is bool,
            'Explicit CFR compatibility flag required')
    require('clock_policy' not in clock or isinstance(clock['clock_policy'], str), 'Clock policy must be text')


def _clock_sha(clock):
    encoded = json.dumps({k: clock[k] for k in CLOCK_KEYS}, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_approved_scope(full_clock, approved_seconds=None, expected_frames=None):
    
    _clock(full_clock)
    total = full_clock['frame_count']
    if approved_seconds is None:
        require(expected_frames is None, 'Cannot infer approval from frame indices')
        seconds, start, end, kind = None, 0, total, 'full_source'
    else:
        seconds = _seconds(approved_seconds)
        require(isinstance(expected_frames, (list, tuple)) and len(expected_frames) == 2
                and all(type(v) is int for v in expected_frames)
                and 0 <= expected_frames[0] < expected_frames[1] <= total,
                'Explicit in-bounds original half-open frame range required')
        start, end = approved_frame_interval(full_clock, seconds)
        require((start, end) == tuple(expected_frames), 'Approved seconds/frame range disagree')
        kind = 'previously_approved_interval'
    selected = copy.deepcopy(full_clock)
    selected.update(frame_count=end-start, source_pts=full_clock['source_pts'][start:end].copy(),
                    source_time_s=full_clock['source_time_s'][start:end].copy())
    scope = dict(version=VERSION, kind=kind, approved_seconds=seconds,
        source_frame_count=total, source_start_frame=start, source_end_frame_exclusive=end,
        input_frame_count=end-start, selection_rule='complete_original_source' if seconds is None
            else 'legacy_floor_ceil_original_fps', input_context_policy='scope_only_no_outside_frames',
        input_clock_sha256=_clock_sha(selected))
    return scope, selected


def validate_scoped_observation(obs, source, scope, input_clock, rgb=None):
    
    _clock(input_clock)
    require(type(source) is dict and set(source) == SOURCE_KEYS,
            'Strict source-only descriptor required; no GT/pair payloads')
    require(isinstance(source['sample_id'], str) and bool(source['sample_id'])
            and isinstance(source['video_path'], str) and Path(source['video_path']).is_absolute()
            and _sha_string(source['video_sha256']), 'Bound absolute original source identity required')
    require(type(obs) is dict and OBS_KEYS <= set(obs) <= OBS_KEYS | OBS_OPTIONAL,
            'Strict source-only observation whitelist required; no GT/pair payloads')
    require(obs['video_path'] == source['video_path'] and obs['video_sha256'] == source['video_sha256']
            and ('sample_id' not in obs or obs['sample_id'] == source['sample_id']), 'Different original source identity')
    require(_sha_string(obs['rgb_sha256']) and ('rgb_npy_sha256' not in obs or _sha_string(obs['rgb_npy_sha256'])),
            'Bound decoded RGB hash required')
    require(all(obs[k] is False for k in FALSE_FLAGS), 'Transformed or annotated observation forbidden')
    _clock({k: obs[k] for k in CLOCK_KEYS})
    require(all(obs[k] == input_clock[k] for k in CLOCK_KEYS), 'Different original input clock/geometry')
    if scope is None:
        require(source['approved_seconds'] is None and obs['approved_seconds'] is None,
                'Explicit approved scope required for an interval')
        scope, _ = build_approved_scope(input_clock)
    require(type(scope) is dict and set(scope) == SCOPE_KEYS and scope['version'] == VERSION
            and scope['input_context_policy'] == 'scope_only_no_outside_frames', 'Explicit immutable source scope required')
    require(all(type(scope[k]) is int for k in ('source_frame_count', 'source_start_frame',
            'source_end_frame_exclusive', 'input_frame_count')), 'Integer original scope indices required')
    total, start, end = (scope[k] for k in ('source_frame_count', 'source_start_frame', 'source_end_frame_exclusive'))
    require(0 <= start < end <= total and end-start == scope['input_frame_count'] == input_clock['frame_count']
            and scope['input_clock_sha256'] == _clock_sha(input_clock), 'Scoped original clock/range binding differs')
    if scope['kind'] == 'full_source':
        require(scope['approved_seconds'] is None and start == 0 and end == total
                and scope['selection_rule'] == 'complete_original_source', 'Full source contract differs')
    else:
        require(scope['kind'] == 'previously_approved_interval'
                and scope['selection_rule'] == 'legacy_floor_ceil_original_fps', 'Unsupported interval convention')
        seconds = _seconds(scope['approved_seconds'])
        require(approved_frame_interval(dict(frame_count=total, fps=input_clock['fps']), seconds) == (start, end),
                'Approved seconds/frame range disagree')
    for value in (source['approved_seconds'], obs['approved_seconds']):
        if value is not None:
            _seconds(value)
        require(value == scope['approved_seconds'], 'Source/observer approved interval differs')
    require(all(type(obs[k]) is int and obs[k] == scope[k] for k in
            ('source_frame_count', 'source_start_frame', 'source_end_frame_exclusive')),
            'Full-cache reuse or incorrect scoped source range')
    if rgb is not None:
        require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8
                and rgb.shape == (input_clock['frame_count'], input_clock['height'], input_clock['width'], 3),
                'Only all selected original RGB frames/native coordinates allowed')
