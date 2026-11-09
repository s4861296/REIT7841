
import io
import json
import math
import re
from fractions import Fraction

import numpy as np


SCHEMA = 'native_flow_transition_v1'
META_FIELDS = frozenset((
    'source_video_sha256', 'frame_index', 'frame_count', 'height', 'width',
    'fps', 'previous_pts_s', 'current_pts_s',
    'previous_pts', 'current_pts', 'time_base',
))
ARRAY_FIELDS = frozenset(('metadata', 'aligned_residual', 'sampling_bits', 'roundtrip_bits'))


def validate_metadata(metadata):
    if not isinstance(metadata, dict) or set(metadata) != META_FIELDS:
        raise ValueError('Require exact label-blind source metadata')
    if not isinstance(metadata['source_video_sha256'], str) or not re.fullmatch(
            r'[0-9a-f]{64}', metadata['source_video_sha256']):
        raise ValueError('Require source SHA256 binding')
    for key in ('frame_index', 'frame_count', 'height', 'width', 'previous_pts', 'current_pts'):
        if type(metadata[key]) is not int:
            raise ValueError('Integer native coordinates required')
    if not (1 <= metadata['frame_index'] < metadata['frame_count']
            and metadata['height'] > 0 and metadata['width'] > 0):
        raise ValueError('Invalid source coordinates or missing predecessor')
    for key in ('fps', 'previous_pts_s', 'current_pts_s'):
        if type(metadata[key]) not in (int, float) or not math.isfinite(metadata[key]):
            raise ValueError('Finite source FPS/PTS required')
    delta = metadata['current_pts_s'] - metadata['previous_pts_s']
    if metadata['fps'] <= 0 or delta <= 0 or not math.isclose(
            delta, 1 / metadata['fps'], rel_tol=0, abs_tol=1e-4):
        raise ValueError('Original adjacent PTS/FPS mismatch')
    if not isinstance(metadata['time_base'], str):
        raise ValueError('Require original rational time base')
    try:
        time_base = Fraction(metadata['time_base'])
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError('Invalid source time base') from exc
    if time_base <= 0 or metadata['current_pts'] <= metadata['previous_pts']:
        raise ValueError('Nonmonotonic integer PTS')
    for which in ('previous', 'current'):
        if not math.isclose(float(metadata[which+'_pts'] * time_base),
                            metadata[which+'_pts_s'], rel_tol=0, abs_tol=1e-12):
            raise ValueError('Integer PTS and absolute seconds disagree')


def _validate_maps(aligned, sampling, roundtrip, metadata):
    shape = (metadata['height'], metadata['width'])
    if not isinstance(aligned, np.ndarray) or aligned.shape != shape or aligned.dtype != np.float32:
        raise ValueError('Require native FP32 aligned residual, no quantization')
    if not np.isfinite(aligned).all() or (aligned < 0).any() or (aligned > 1).any():
        raise ValueError('Residual outside finite normalized RGB range')
    for value in (sampling, roundtrip):
        if not isinstance(value, np.ndarray) or value.dtype != np.bool_ or value.shape != shape:
            raise ValueError('Require native boolean geometry support')
    if np.any(roundtrip & ~sampling):
        raise ValueError('Roundtrip support must be a subset of sampling support')


def encode_transition(aligned_residual, sampling_support, roundtrip_support, metadata):
    
    validate_metadata(metadata)
    _validate_maps(aligned_residual, sampling_support, roundtrip_support, metadata)
    header = json.dumps({'schema': SCHEMA, 'source': metadata}, sort_keys=True,
                        separators=(',', ':'), allow_nan=False).encode('utf-8')
    stream = io.BytesIO()
    np.savez_compressed(
        stream, metadata=np.frombuffer(header, dtype=np.uint8),
        aligned_residual=aligned_residual,
        sampling_bits=np.packbits(sampling_support.ravel(order='C'), bitorder='little'),
        roundtrip_bits=np.packbits(roundtrip_support.ravel(order='C'), bitorder='little'),
    )
    return stream.getvalue()


def decode_transition(payload, expected_metadata):
    
    validate_metadata(expected_metadata)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if len(archive.files) != len(ARRAY_FIELDS) or set(archive.files) != ARRAY_FIELDS:
            raise ValueError('Unexpected cache members')
        header = archive['metadata']
        if header.dtype != np.uint8 or header.ndim != 1:
            raise ValueError('Invalid metadata encoding')
        obj = json.loads(header.tobytes().decode('utf-8'))
        if set(obj) != {'schema', 'source'} or obj['schema'] != SCHEMA:
            raise ValueError('Wrong cache schema')
        validate_metadata(obj['source'])
        if obj['source'] != expected_metadata:
            raise ValueError('Cache belongs to different source/frame/PTS/grid')
        h, w = obj['source']['height'], obj['source']['width']
        size = h * w

        def unpack(name):
            packed = archive[name]
            if packed.dtype != np.uint8 or packed.shape != ((size + 7) // 8,):
                raise ValueError('Invalid packed native support length')
            all_bits = np.unpackbits(packed, bitorder='little')
            if all_bits[size:].any():
                raise ValueError('Nonzero unused support padding bits')
            return all_bits[:size].reshape(h, w).astype(np.bool_)

        aligned = archive['aligned_residual']
        sampling, roundtrip = unpack('sampling_bits'), unpack('roundtrip_bits')
    _validate_maps(aligned, sampling, roundtrip, expected_metadata)
    return {'aligned_residual': aligned, 'sampling_support': sampling,
            'roundtrip_support': roundtrip, 'metadata': obj['source']}
