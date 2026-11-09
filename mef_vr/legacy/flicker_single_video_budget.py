
from pathlib import Path
import shutil

GIB = 1024**3
MIB = 1024**2
RESERVE_BYTES = 40 * GIB


def _integer(value, name, *, positive=True):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(name+' must be a '+('positive' if positive else 'nonnegative')+' integer')


def estimate_single_video_disk(frames, height, width, *, upstream_window, upstream_stride,
                               approved_start=0, approved_stop=None, chunk_frames=16):
    
    for name, value in (('frames', frames), ('height', height), ('width', width),
                        ('upstream_window', upstream_window), ('upstream_stride', upstream_stride),
                        ('chunk_frames', chunk_frames)):
        _integer(value, name)
    _integer(approved_start, 'approved_start', positive=False)
    stop = frames if approved_stop is None else approved_stop
    _integer(stop, 'approved_stop')
    if not 0 <= approved_start < stop <= frames:
        raise ValueError('Approved original frame interval must lie inside the source')
    if upstream_stride > upstream_window:
        raise ValueError('Upstream stride would leave original frames uncovered')
    n = stop-approved_start; gh, gw = (height+13)//14, (width+13)//14
    grid = gh*gw; native = height*width; packed_per_frame = (native+7)//8
    chunks = (n+chunk_frames-1)//chunk_frames
    windows = 1 if n <= upstream_window else (n-upstream_window)//upstream_stride+1+int((n-upstream_window) % upstream_stride != 0)
    components = {}

    def add(name, size, explanation, kind='retained_allowance'):
        components[name] = dict(bytes=size, kind=kind, explanation=explanation)
        return size

    rgb = add('rgb_uint8_npy_payload', n*native*3, 'source/rgb.npy: every retained native RGB frame, uint8 T,H,W,3')
    semantic = add('dino_fp16_chunk_payloads', n*384*grid*2,
        'semantic_chunks/*.npy retained: T,384,ceil(H/14),ceil(W/14), float16; no duplicated on-disk concatenation')
    motion = add('raft_pooled_fp32_chunk_payloads', n*5*grid*4,
        'motion_chunks/*.npy retained: only five pooled float32 channels; native dense RAFT flow is NOT saved')
    feature = add('features51_fp16_npy_payload', n*51*grid*2, 'features_CPU.npy: complete 51-channel float16 grid')
    add('upstream_h_window_fp32_payloads', windows*upstream_window*4,
        'H_windows/*.npy: each complete upstream probability window, including its padded short-source length')
    native_npz = add('native_C_npz_uncompressed_payload', n*(packed_per_frame+4+8),
        'native_C.npz: per-frame ceil(H*W/8) mask bytes + float32 maxima + float64 time; no compression savings assumed')
    final_npz = add('final_mask_npz_uncompressed_payload', n*(packed_per_frame+8+8)+3*8,
        'predicted_masks.npz: packed masks, int64 original PTS, float64 times, int64 native shape')
    heads_npz = add('two_head_npz_uncompressed_payload', n*(2*(1+2+1+2)*4+4+1+8+8)+64,
        'Two heads: state/boundary logits + probabilities FP32; final state/flag and original int64 PTS/float64 time')
    array_files = 2+2*chunks+windows
    add('array_headers_and_npz_expansion', array_files*4096+3*65536+(native_npz+final_npz+heads_npz+49)//50,
        '4KiB per NPY; 64KiB per NPZ plus 2% NPZ payload expansion, not an assumed compression ratio')
    metadata_files = 2*chunks+windows+32
    add('json_clock_sidecars_and_receipts', 16*MIB+n*4096+metadata_files*(65536+64*n),
        '16MiB fixed + 4KiB/frame + 64KiB/file and 64 bytes/frame/file for repeated original-PTS contracts and sidecars')
    review = add('review_encoded_video_allowance', 4*rgb+n*65536+16*MIB,
        'Separate same-native-size libx264 review: 4x RGB bytes + 64KiB/frame + 16MiB mux allowance; no bitrate guarantee')
    add('frozen_code_tests_and_logs', 16*MIB,
        'Code/test snapshot, test logs, small entry receipts; pretrained weight files are bound in place, not copied')
    peak_chunk = min(n, chunk_frames)*grid*max(384*2, 5*4)+4096
    add('one_extra_staged_or_partial_artifact', max(rgb+4096, feature+4096, peak_chunk,
                                                native_npz+65536, final_npz+65536, heads_npz+65536, review),
        'One extra largest artifact allowance for staging/retained partial; retries must recheck full free-space admission',
        'temporary_or_preserved_partial_allowance')
    add('in_memory_concatenation_disk_bytes', 0,
        'np.concatenate of DINO/RAFT chunks and feature halo/.copy() are RAM-only in the reused implementation; RAM is NOT budgeted',
        'not_written_to_disk')
    subtotal = sum(c['bytes'] for c in components.values())
    add('estimation_uncertainty_margin', (subtotal+3)//4,
        'Explicit 25% margin for serialization, codec and metadata estimate error; not a proven worst-case codec bound',
        'estimation_margin')
    additional = sum(c['bytes'] for c in components.values())
    return dict(version='single_video_disk_budget_v1', original_frames=frames,
        approved_start_frame=approved_start, approved_stop_frame_exclusive=stop, retained_frames=n,
        full_original_source=approved_start == 0 and stop == frames, height=height, width=width,
        grid_height=gh, grid_width=gw, packed_mask_bytes_per_frame=packed_per_frame,
        chunk_frames=chunk_frames, chunks_per_observation_family=chunks,
        upstream_window=upstream_window, upstream_stride=upstream_stride, upstream_window_count=windows,
        components=components, required_additional_bytes=additional, reserve_bytes=RESERVE_BYTES,
        required_free_bytes=additional+RESERVE_BYTES, required_additional_gib=additional/GIB,
        required_free_gib=(additional+RESERVE_BYTES)/GIB,
        existing_occupied_bytes_deducted=0, source_video_copy_bytes=0, pretrained_weight_copy_bytes=0,
        dense_raft_flow_saved=False, memory_or_vram_budget=False, source_resize_or_resampling=False,
        original_source_reencoded=False, outputs_or_partials_deleted=False,
        limitations=['Geometry/range must be validated against the original clock by the caller; no source is read here.',
            'Review encoding and JSON sizes are conservative engineering allowances, not mathematical hard upper bounds.',
            'Check admission before work/retry and enforce the 40GiB floor at output stages; never delete old artifacts to pass.',
            'Does not cover copied model weights, additional reviews or unrelated/concurrent writers; no existing-byte credit.'])


def check_disk_budget(path, estimate):
    
    required = estimate['required_additional_bytes']; reserve = estimate['reserve_bytes']
    _integer(required, 'required_additional_bytes', positive=False)
    if reserve != RESERVE_BYTES or estimate['required_free_bytes'] != required+RESERVE_BYTES:
        raise ValueError('Budget/reserve mismatch')
    free = shutil.disk_usage(Path(path)).free
    if free < required+reserve:
        raise OSError(f'Need {required+reserve} free bytes, including unchanged 40GiB reserve; have {free}. Keep old files.')
    return dict(free_bytes=free, required_free_bytes=required+reserve, sufficient=True,
                expected_free_after_budget_bytes=free-required, reserve_bytes=reserve)
