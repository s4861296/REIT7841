
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
from fractions import Fraction
import zipfile
import numpy as np
from native_flow_cache_codec import validate_metadata


SCHEMA = 'native_motion_sequence_v1'
SOURCE_KEYS = {'source_video_sha256', 'frame_count', 'height', 'width', 'fps', 'pts', 'time_base'}
PROVENANCE_KEYS = {'algorithm', 'weights_sha256', 'implementation_sha256', 'iterations',
                   'flow_dtype', 'consistency_alpha', 'consistency_beta'}


def _sha_bytes(data): return hashlib.sha256(data).hexdigest()


def _hash_string(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def validate_source(source):
    if not isinstance(source, dict) or set(source) != SOURCE_KEYS:
        raise ValueError('Exact label-blind source fields required')
    if not _hash_string(source['source_video_sha256']): raise ValueError('Source SHA required')
    for key in ('frame_count', 'height', 'width'):
        if type(source[key]) is not int or source[key] < 1: raise ValueError('Native positive integer dimensions required')
    if type(source['fps']) not in (int, float) or not math.isfinite(source['fps']) or source['fps'] <= 0:
        raise ValueError('Original FPS required')
    if (not isinstance(source['pts'], list) or len(source['pts']) != source['frame_count']
            or any(type(t) is not int for t in source['pts'])):
        raise ValueError('One original integer PTS per frame required')
    if not isinstance(source['time_base'], str): raise ValueError('Rational time base required')
    try: base = Fraction(source['time_base'])
    except (ValueError, ZeroDivisionError) as error: raise ValueError('Invalid time base') from error
    if base <= 0: raise ValueError('Positive time base required')
    for index in range(1, source['frame_count']): validate_metadata(transition_metadata(source, index))


def validate_provenance(provenance):
    if not isinstance(provenance, dict) or set(provenance) != PROVENANCE_KEYS:
        raise ValueError('Exact label-blind frozen producer provenance required')
    if provenance['algorithm'] != 'torchvision_raft_large_bidirectional': raise ValueError('Unexpected flow producer')
    if not all(_hash_string(provenance[k]) for k in ('weights_sha256', 'implementation_sha256')):
        raise ValueError('Producer hashes required')
    if type(provenance['iterations']) is not int or provenance['iterations'] < 1:
        raise ValueError('Explicit iteration count required')
    if provenance['flow_dtype'] != 'float32': raise ValueError('Unquantized flow required')
    if (type(provenance['consistency_alpha']) not in (int, float)
            or type(provenance['consistency_beta']) not in (int, float)
            or provenance['consistency_alpha'] != .01 or provenance['consistency_beta'] != .5):
        raise ValueError('Fixed diagnostic consistency criterion required')


def transition_metadata(source, index):
    if type(index) is not int or not 1 <= index < source['frame_count']:
        raise ValueError('Frame zero has no transition; source range required')
    base = Fraction(source['time_base'])
    return {k: source[k] for k in ('source_video_sha256', 'frame_count', 'height', 'width', 'fps', 'time_base')} | {
        'frame_index': index, 'previous_pts': source['pts'][index-1], 'current_pts': source['pts'][index],
        'previous_pts_s': float(source['pts'][index-1]*base), 'current_pts_s': float(source['pts'][index]*base)}


def _validate_arrays(aligned, support, consistent, metadata):
    validate_metadata(metadata)
    shape = (metadata['height'], metadata['width'])
    if (not isinstance(aligned, np.ndarray) or aligned.shape != shape or aligned.dtype != np.float32
            or not np.isfinite(aligned).all() or (aligned < 0).any() or (aligned > 1).any()):
        raise ValueError('Finite native normalized FP32 residual required')
    for value in (support, consistent):
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != np.bool_:
            raise ValueError('Original-grid boolean maps required')
    if np.any(consistent & ~support): raise ValueError('Consistency outside support is untested')


def encode_motion_transition(aligned, support, consistent, metadata):
    _validate_arrays(aligned, support, consistent, metadata)
    head = json.dumps({'schema': SCHEMA, 'source': metadata}, sort_keys=True, allow_nan=False).encode()
    out = io.BytesIO()
    np.savez_compressed(out, metadata=np.frombuffer(head, np.uint8), aligned_residual=aligned,
                        sampling_bits=np.packbits(support.ravel(), bitorder='little'),
                        consistency_bits=np.packbits(consistent.ravel(), bitorder='little'))
    return out.getvalue()


def decode_motion_transition(payload, metadata):
    validate_metadata(metadata)
    with np.load(io.BytesIO(payload), allow_pickle=False) as z:
        names = {'metadata', 'aligned_residual', 'sampling_bits', 'consistency_bits'}
        if len(z.files) != len(names) or set(z.files) != names: raise ValueError('Unexpected transition members')
        header = z['metadata']
        if header.dtype != np.uint8 or header.ndim != 1: raise ValueError('Invalid transition header')
        if json.loads(header.tobytes()) != {'schema': SCHEMA, 'source': metadata}:
            raise ValueError('Source/frame/PTS or schema mismatch')
        h, w = metadata['height'], metadata['width']; pixels = h*w
        def unpack(name):
            bits = z[name]
            if bits.dtype != np.uint8 or bits.shape != ((pixels+7)//8,): raise ValueError('Invalid bitmap length')
            raw = np.unpackbits(bits, bitorder='little')
            if raw[pixels:].any(): raise ValueError('Nonzero unused padding')
            return raw[:pixels].reshape(h, w).astype(bool)
        aligned = z['aligned_residual']; support = unpack('sampling_bits'); consistent = unpack('consistency_bits')
    _validate_arrays(aligned, support, consistent, metadata)
    return aligned, support, consistent


def write_sequence(path, source, provenance, transitions, *, max_bytes, min_free_bytes):
    
    validate_source(source); validate_provenance(provenance)
    if type(max_bytes) is not int or max_bytes <= 0 or type(min_free_bytes) is not int or min_free_bytes < 0:
        raise ValueError('Explicit positive byte budget and disk reserve required')
    path = Path(path); partial = path.with_name(path.name+'.partial')
    if path.exists() or partial.exists() or path.is_symlink() or partial.is_symlink():
        raise FileExistsError('Never overwrite a prior cache or failed attempt')
    if not path.parent.is_dir(): raise ValueError('Caller must create the explicit output directory')
    entries = []; source = json.loads(json.dumps(source)); provenance = dict(provenance)
    transitions = iter(transitions)
    def check_budget(current, addition, final=False):
        allowance = 0 if final else 4096*(len(entries)+2)
        if current+addition+allowance > max_bytes: raise OSError('Per-video byte cap reached')
        if shutil.disk_usage(path.parent).free < min_free_bytes+addition+allowance:
            raise OSError('Disk reserve reached; preserve old data')
    check_budget(0, 0)
    with partial.open('xb') as stream, zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for expected_index in range(1, source['frame_count']):
            try: index, aligned, support, consistent = next(transitions)
            except StopIteration as error: raise ValueError('Incomplete full-video transition sequence') from error
            if type(index) is not int or index != expected_index: raise ValueError('Missing, repeated or reordered frame')
            metadata = transition_metadata(source, index)
            payload = encode_motion_transition(aligned, support, consistent, metadata)
            decoded = decode_motion_transition(payload, metadata)
            if any(a.tobytes() != b.tobytes() for a, b in zip((aligned, support, consistent), decoded)):
                raise ValueError('Non-lossless native cache')
            check_budget(stream.tell(), len(payload))
            name = f'frames/{index:06d}.npz'; archive.writestr(name, payload)
            entries.append({'frame_index': index, 'member': name, 'sha256': _sha_bytes(payload), 'bytes': len(payload)})
        try: next(transitions)
        except StopIteration: pass
        else: raise ValueError('Extra transitions outside original video')
        header = {'schema': SCHEMA, 'complete': True, 'source': source, 'producer': provenance,
                  'frame_zero': 'missing_predecessor_explicit_zero_cues', 'transitions': entries}
        payload = json.dumps(header, sort_keys=True, allow_nan=False).encode()
        check_budget(stream.tell(), len(payload)); archive.writestr('manifest.json', payload)
    check_budget(partial.stat().st_size, 0, final=True)
    
    import os
    os.link(partial, path)
    partial.unlink() 
    return header


class MotionSequenceReader:
    def __init__(self, path, source, provenance):
        validate_source(source); validate_provenance(provenance)
        if Path(path).name.endswith('.partial'): raise ValueError('Uncommitted cache cannot enter training')
        self.source = json.loads(json.dumps(source)); self.archive = zipfile.ZipFile(path, 'r')
        try:
            members = self.archive.namelist()
            if len(set(members)) != len(members): raise ValueError('Duplicate archive member')
            header = json.loads(self.archive.read('manifest.json'))
            if (set(header) != {'schema', 'complete', 'source', 'producer', 'frame_zero', 'transitions'}
                    or header['schema'] != SCHEMA or header['complete'] is not True
                    or header['source'] != source or header['producer'] != provenance
                    or header['frame_zero'] != 'missing_predecessor_explicit_zero_cues'):
                raise ValueError('Incomplete or mismatched sequence provenance')
            entries = header['transitions']
            if not isinstance(entries, list) or len(entries) != source['frame_count']-1:
                raise ValueError('Incomplete original frame coverage')
            for index, entry in enumerate(entries, 1):
                if (set(entry) != {'frame_index', 'member', 'sha256', 'bytes'}
                        or type(entry['frame_index']) is not int or entry['frame_index'] != index
                        or entry['member'] != f'frames/{index:06d}.npz'
                        or not _hash_string(entry['sha256']) or type(entry['bytes']) is not int or entry['bytes'] <= 0):
                    raise ValueError('Invalid transition coverage or hash binding')
            if set(members) != {'manifest.json'} | {e['member'] for e in entries}:
                raise ValueError('Missing/extra transition files')
            self.entries = entries
        except BaseException:
            self.archive.close(); raise

    def close(self): self.archive.close()
    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def chunk(self, left, right):
        
        if type(left) is not int or type(right) is not int or not 0 <= left < right <= self.source['frame_count']:
            raise ValueError('Exact in-range integer source interval required')
        h, w = self.source['height'], self.source['width']
        cues = np.zeros((right-left, 4, h, w), np.float32)
        for index in range(left, right):
            if index == 0: continue
            entry = self.entries[index-1]; payload = self.archive.read(entry['member'])
            if len(payload) != entry['bytes'] or _sha_bytes(payload) != entry['sha256']:
                raise ValueError('Transition content corrupt or replaced')
            aligned, support, consistent = decode_motion_transition(payload, transition_metadata(self.source, index))
            cues[index-left, 0] = aligned*support
            cues[index-left, 1] = support; cues[index-left, 2] = consistent; cues[index-left, 3] = 1
        return cues
