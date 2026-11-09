
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

import numpy as np

VERSION = 'new_train_first_pair_frozen_gpu_observations_v1'
MEDIA = Path('/mnt/e/VRFlickerLab/model_evaluation/new_train_temporal_candidate_media_20261011_v1')
MEDIA_COMPLETED_SHA = '4999ad83b68fdfa168bef950a5d8d9fd2ddaa3b7c7a3ecac8a20d095da2ecb84'
MEDIA_REPORT_SHA = 'bc647147c706f891fe3970a5b95a84a59c214211bf4afc20fd9e0e502d463fe9'
MEDIA_SOURCES_SHA = 'a7c957a9764e2393ae92d72ce0d7a870065fd6ba219953edee2438c4b363b4af'
PARENT = Path('/mnt/e/VRFlickerLab/model_evaluation/multimodel_historical_real_21_20261007_v1')
PARENT_SHA = '18232016eeb0ff6ec48b5e931cdb56b5b9c087d8aa10274eea9b12dea5339ae8'
PAIR = 'composition_descriptor_cache_overflow::moving_camera::QV5CM__displayxr_389_vk_descriptor_cache_exhaustion__032'
NAME = 'prepare_flicker_new_train_gpu_observations.py'
TEST = 'test_flicker_new_train_gpu_observations.py'
CLOCK_KEYS = ('frame_count', 'height', 'width', 'fps', 'fps_fraction', 'time_base_fraction', 'source_pts', 'source_time_s')
SOURCE_FILES = ('source_started.json', 'source/observation.json', 'source/status.json', 'source/rgb.npy',
                'upstream.json', 'native_C.npz', 'motion_pooled.npy', 'motion_pooled.json', 'features.npy', 'features.json')
CONFIG = dict(upstream_window=32, upstream_stride=8, upstream_difference_context='window_zero',
              upstream_feature_mode='appearance_delta_mean_max', upstream_dense_difference_branch='shallow_native',
              upstream_head_channels=64, feature_core=32, feature_halo=8)


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def read(path, digest=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Regular bound JSON required: ' + str(path))
    data = path.read_bytes()
    if digest is not None: require(hashlib.sha256(data).hexdigest() == digest, 'JSON hash differs: ' + str(path))
    return json.loads(data)


def publish(path, value):
    
    path = Path(path)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def utc(): return datetime.now(timezone.utc).isoformat()


def feature_names():
    names = [f'dino_projection_{i}' for i in range(16)]
    for lag in (1, 4, 8):
        for direction in (-1, 1): names += [f'dino_cosine_{direction * lag}', f'dino_available_{direction * lag}']
    names += ['rgb_mean_r', 'rgb_mean_g', 'rgb_mean_b', 'rgb_std_r', 'rgb_std_g', 'rgb_std_b',
              'raw_previous_mean', 'raw_previous_max', 'raw_next_mean', 'raw_next_max',
              'raft_aligned_mean', 'raft_support', 'raft_consistent', 'raft_available', 'raft_aligned_max',
              'frozen_C_mask_fraction']
    names += [f'rgb_patch_difference_{direction * lag}' for lag in (1, 4, 8) for direction in (-1, 1)]
    return names + ['frozen_r3d_context']


def validate_clock(clock):
    require(all(k in clock for k in CLOCK_KEYS), 'Incomplete original clock')
    n = clock['frame_count']; pts = clock['source_pts']; times = clock['source_time_s']
    require(all(type(clock[k]) is int and clock[k] > 0 for k in ('frame_count', 'height', 'width')),
            'Positive integer native geometry required')
    require(len(pts) == n and len(times) == n and all(type(v) is int for v in pts), 'Incomplete original PTS')
    base, rate = Fraction(clock['time_base_fraction']), Fraction(clock['fps_fraction'])
    require(base > 0 and rate > 0 and clock['fps'] == float(rate), 'FPS/time base differ')
    require(all(b > a for a, b in zip(pts, pts[1:])), 'Nonincreasing original PTS')
    require(all(np.isfinite(t) and t == float(p * base) for p, t in zip(pts, times)), 'PTS/time mismatch')


def select_pair(rows, pairs):
    require(len(rows) == 22 and len({r['sample_id'] for r in rows}) == 22, 'Expected22 unique validated sources')
    chosen = sorted((r for r in rows if r['pair_id'] == PAIR), key=lambda r: r['role'])
    require(len(chosen) == 2 and {r['role'] for r in chosen} == {'bug', 'clean_or_fixed'}, 'First full pair only')
    require({r['sample_id'] for r in chosen} == {PAIR+'::bug', PAIR+'::clean_or_fixed'}, 'Pair/source identity differs')
    pair_rows = [r for r in pairs if r['pair_id'] == PAIR]
    require(len(pair_rows) == 1 and pair_rows[0]['identical_complete_original_clock'] is True
            and set(pair_rows[0]['source_ids']) == {r['sample_id'] for r in chosen}, 'Missing bound pair clock')
    sources, mapping = [], []
    for i, r in enumerate(chosen):
        clock = {k: r['clock'][k] for k in CLOCK_KEYS}; validate_clock(clock)
        require((clock['frame_count'], clock['height'], clock['width'], clock['fps_fraction']) == (180, 720, 1280, '60'),
                'First pair must retain all180 original720p60FPS frames')
        require(r['clock_and_metadata_passed'] is True and r['all_original_frames_preserved'] is True
                and r['frame_drop'] is False and r['spatial_resize'] is False and r['fps_resampled'] is False
                and not r['current_inventory_content_hash_matches']
                and r['content_duplicate_of_current_train'] is False and r['content_duplicate_of_current_development'] is False,
                'Validated source eligibility differs')
        require(r['rgb_decoded_frames'] == r['scan_decoded_frames'] == clock['frame_count'], 'Decode count differs')
        key = f'source_{i:03d}'
        sources.append(dict(source_key=key, video_path=r['video_path'], video_sha256=r['video_sha256'],
                            expected_rgb_sha256=r['rgb_stream_sha256'], clock=clock, approved_seconds=None,
                            video_hash_provenance=r['hash_provenance']))
        mapping.append(dict(source_key=key, sample_id=r['sample_id'], pair_id=r['pair_id'], role=r['role'],
                            scene_id=r['scene_id'], exposure='upstream-exposed existing Train scene; NOT fresh holdout'))
    require(sources[0]['clock'] == sources[1]['clock'], 'Bug/Clean original clocks differ')
    require(sources[0]['video_sha256'] != sources[1]['video_sha256']
            and sources[0]['expected_rgb_sha256'] != sources[1]['expected_rgb_sha256'], 'Identical Bug/Clean source')
    return sources, mapping


def source_plan():
    done = read(MEDIA/'completed.json', MEDIA_COMPLETED_SHA)
    require(done['completed'] is True and done['media_preparation_passed'] is True, 'Media preparation incomplete')
    require(done['files_sha256']['report.json'] == MEDIA_REPORT_SHA
            and done['files_sha256']['sources.json'] == MEDIA_SOURCES_SHA, 'Media completion/report binding differs')
    report = read(MEDIA/'report.json', MEDIA_REPORT_SHA)
    require(report['source_count'] == 22 and report['pair_count'] == 11 and report['original_frames'] == 3600
            and report['media_preparation_passed'] is True and report['current_content_hash_matching_sources'] == 0,
            'Fixed media scope differs')
    sources, mapping = select_pair(read(MEDIA/'sources.json', MEDIA_SOURCES_SHA),
                                   read(MEDIA/'pairs.json', done['files_sha256']['pairs.json']))
    bindings = {str(MEDIA/name): digest for name, digest in (
        ('completed.json', MEDIA_COMPLETED_SHA), ('report.json', MEDIA_REPORT_SHA),
        ('sources.json', MEDIA_SOURCES_SHA), ('pairs.json', done['files_sha256']['pairs.json']))}
    return sources, mapping, bindings


def destination(value):
    out = Path(value).absolute()
    root = Path('/mnt/e/VRFlickerLab')
    require(root in out.parents and out.parent.is_dir() and out.resolve() == out and not out.is_symlink(),
            'Fresh non-symlink persistent E/VRFlickerLab directory required')
    return out


def prepare(out):
    out = destination(out)
    if out.exists(): raise FileExistsError('Existing output preserved')
    sources, mapping, bindings = source_plan()
    original = read(PARENT/'protocol.json', PARENT_SHA)
    require(all(original[k] == v for k, v in CONFIG.items()), 'Frozen GPU settings differ')
    require(set(original['models']) == {'backbone_selection', 'backbone', 'head', 'native_C', 'raft', 'dino'},
            'Original frozen models differ')
    require(original['raft_producer']['iterations'] == 12, 'Original RAFT iterations differ')
    
    require(all(Path(path).is_file() and not Path(path).is_symlink() for path in original['model_sha256']),
            'Missing frozen model/config; no downloading/substitution')
    for name, digest in original['code_sha256'].items():
        require(Path(name).name == name and sha(PARENT/'code'/name) == digest, 'Frozen producer code differs: '+name)
    require(shutil.disk_usage(out.parent).free >= 45 * 1024**3, 'Keep40GiB plus first-pair derived storage')
    out.mkdir(); (out/'code').mkdir()
    for name in original['code_sha256']: shutil.copyfile(PARENT/'code'/name, out/'code'/name)
    shutil.copyfile(Path(__file__), out/'code'/NAME)
    shutil.copyfile(Path(__file__).resolve().parent.parent/'tests'/TEST, out/TEST)
    publish(out/'sources_observed.json', sources); publish(out/'source_mapping_not_model_input.json', mapping)
    publish(out/'producer_protocol.json', original)
    p = dict(version=VERSION, prepared_at=utc(), pair_id=PAIR, sources=2, original_frames=360,
             parent_protocol_path=str(PARENT/'protocol.json'), parent_protocol_sha256=PARENT_SHA,
             input_bindings=bindings, models=original['models'], model_sha256=original['model_sha256'],
             raft_producer=original['raft_producer'], **CONFIG,
             device='cuda:0', threads=2, opencv_threads=1, cuda_memory_fraction=.65,
             minimum_free_gpu_gib=20, reserve_disk_gib=40, seed=20261011,
             cuda_matmul_tf32=False, cudnn_benchmark=False, cublas_workspace_config=':4096:8',
             stage_backend='Original GPU observers; H BF16; C/RAFT FP32; DINO BF16 then FP16; original stage cudnn flags',
             feature_channels=51, expected_feature_names=feature_names(), no_source_resize=True,
             all_original_frames=True, original_fps_pts=True, raw51_not_normalized=True,
             gt_or_paired_clean_in_forward=False, source_identity_or_label_in_forward=False,
             source_identities_used_only_for_preparation=True, gt_mask_files_never_opened=True,
             training_updates=0, downstream_temporal_inference=False, automatic_training_authorized=False,
             upstream_exposure=True, new_holdout=False, visual_clean_quality_claimed=False,
             numeric_equivalence_to_CPU_claimed=False,
             failure_policy='Preserve incomplete source; refuse stage-cache reuse. Resume only completed bound sources.',
             code_sha256={name: sha(out/'code'/name) for name in list(original['code_sha256'])+[NAME]},
             inputs_sha256={name: sha(out/name) for name in ('sources_observed.json', 'source_mapping_not_model_input.json',
                                                           'producer_protocol.json', TEST)})
    publish(out/'protocol.json', p)
    publish(out/'prepared.json', dict(prepared=True, protocol_sha256=sha(out/'protocol.json'),
                                     GPU_started=False, models_loaded=False, training_started=False))
    return p


def validate_observation(observation, source, rgb):
    clock = source['clock']; validate_clock(clock)
    require(all(observation[k] == clock[k] for k in CLOCK_KEYS), 'Decoded original clock differs')
    require(observation['video_sha256'] == source['video_sha256']
            and observation['video_path'] == str(Path(source['video_path']).resolve())
            and observation['rgb_sha256'] == source['expected_rgb_sha256'], 'RGB/source identity differs')
    require(observation['source_frame_count'] == clock['frame_count'] and observation['source_start_frame'] == 0
            and observation['source_end_frame_exclusive'] == clock['frame_count']
            and observation['approved_seconds'] is None, 'Original full source required')
    require(rgb.shape == (clock['frame_count'], clock['height'], clock['width'], 3) and rgb.dtype == np.uint8,
            'Native RGB array differs')
    require(all(observation[k] is False for k in ('spatial_resize', 'temporal_sampling', 'fps_resampled',
                'video_reencoded', 'labels_loaded', 'paired_clean_used')), 'Forbidden source transformation/input')


def validate_outputs(folder, source, upstream, packed, maxima, pooled, features):
    n, h, w = (source['clock'][k] for k in ('frame_count', 'height', 'width')); gh, gw = (h+13)//14, (w+13)//14
    scores = np.asarray(upstream['frame_probabilities']); coverage = np.asarray(upstream['coverage'])
    require(scores.shape == maxima.shape == coverage.shape == (n,) and upstream['labels_used'] is False,
            'Incomplete upstream frame coverage')
    require(np.issubdtype(coverage.dtype, np.integer) and np.all(coverage > 0), 'Unobserved source frames')
    require(all(np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all() for a in (scores, maxima)),
            'Nonfinite/out-of-range prediction')
    require(packed.shape == (n, (h*w+7)//8) and packed.dtype == np.uint8, 'Invalid native predicted-mask geometry')
    if h*w % 8: require(not np.unpackbits(packed[:, -1:], axis=1, bitorder='little')[:, h*w % 8:].any(), 'Nonzero mask padding')
    require(pooled.shape == (n, 5, gh, gw) and pooled.dtype == np.float32
            and np.isfinite(pooled).all() and (pooled >= 0).all() and (pooled <= 1).all(), 'Invalid native RAFT pool')
    require(not pooled[0].any() and np.all(pooled[1:, 3] == 1), 'Missing/invalid predecessor availability')
    require(features.shape == (n, 51, gh, gw) and features.dtype == np.float16, 'Canonical51 native feature shape differs')
    for first in range(0, n, 32): require(np.isfinite(features[first:first+32]).all(), 'Nonfinite feature cache')
    meta = read(folder/'features.json'); motion = read(folder/'motion_pooled.json')
    require(meta['feature_names'] == feature_names() and meta['frames'] == motion['frames'] == n
            and meta['native_source_resolution'] == [h, w] and meta['all_original_frames'] is True
            and meta['gt_used'] is False and motion['gt_used'] is False, 'Feature semantics/coverage differ')
    require(meta['sha256'] == sha(folder/'features.npy') and motion['sha256'] == sha(folder/'motion_pooled.npy'),
            'Committed feature/motion bytes differ')


def extract_source(folder, p, source, components, update):
    
    require(not folder.exists(), 'Partial source retained; no implicit reuse')
    folder.mkdir()
    publish(folder/'source_started.json', dict(source=source, protocol_sha256=p['_protocol_sha256'], started_at=utc()))
    obs = components.create_observed_store(source['video_path'], folder/'source', source['video_sha256'], None)
    rgb = np.load(folder/'source'/'rgb.npy', mmap_mode='r', allow_pickle=False)
    validate_observation(obs, source, rgb)
    observer_config = {k: p[k] for k in (*CONFIG, 'models', 'raft_producer')}
    upstream = components.upstream(folder, observer_config, rgb, update)
    scores = np.asarray(upstream['frame_probabilities'], np.float64)
    packed, maxima = components.native_mask(folder, observer_config, rgb, obs, scores, update)
    pooled = components.motion(folder, observer_config, rgb, update)
    features = components.features(folder, observer_config, rgb, scores, packed, pooled, update)
    validate_outputs(folder, source, upstream, packed, maxima, pooled, features)
    hashes = {name: sha(folder/name) for name in SOURCE_FILES if name != 'source/rgb.npy'}
    
    hashes['source/rgb.npy'] = obs['rgb_npy_sha256']
    result = dict(completed=True, protocol_sha256=p['_protocol_sha256'], source_key=source['source_key'], source=source, files_sha256=hashes,
                  original_frames=obs['frame_count'], feature_shape=list(features.shape),
                  labels_read=False, paired_clean_used=False, training_updates=0)
    publish(folder/'completed.json', result)
    return result


def verify_completed_source(folder, source, protocol_sha256):
    done = read(folder/'completed.json')
    require(done['completed'] is True and done['source'] == source and done['source_key'] == source['source_key']
            and done['protocol_sha256'] == protocol_sha256 and set(done['files_sha256']) == set(SOURCE_FILES),
            'Committed source binding differs')
    for name, digest in done['files_sha256'].items():
        path = folder/name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and path.is_file()
                and not path.is_symlink() and sha(path) == digest, 'Completed source payload differs')
    return done


def verify_prepared(out):
    anchor = read(out/'prepared.json'); p = read(out/'protocol.json', anchor['protocol_sha256'])
    require(anchor['prepared'] is True and p['version'] == VERSION and p['pair_id'] == PAIR
            and p['sources'] == 2 and p['original_frames'] == 360, 'Fixed prepared scope differs')
    require(Path(__file__).resolve() == out/'code'/NAME, 'Run frozen prepared entry only')
    for name, digest in p['code_sha256'].items(): require(sha(out/'code'/name) == digest, 'Frozen code differs: '+name)
    for name, digest in p['inputs_sha256'].items(): require(sha(out/name) == digest, 'Frozen input differs: '+name)
    for path, digest in p['input_bindings'].items(): require(sha(path) == digest, 'Media proof differs: '+path)
    require(sha(PARENT/'protocol.json') == PARENT_SHA == p['parent_protocol_sha256'], 'Parent configuration differs')
    return p, read(out/'sources_observed.json')


def run(out, resume=False):
    out = destination(out); p, sources = verify_prepared(out)
    p['_protocol_sha256'] = sha(out/'protocol.json')
    lock = (out/'worker.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not (out/'completed.json').exists(), 'Completed output preserved')
    if (out/'started.json').exists(): require(resume, 'Explicit resume required')
    else:
        require(not resume, 'Cannot resume an unstarted output')
        publish(out/'started.json', dict(protocol_sha256=sha(out/'protocol.json'), started_at=utc(), pid=os.getpid()))
    for path, digest in p['model_sha256'].items(): require(sha(path) == digest, 'Frozen model differs: '+path)
    for source in sources: require(sha(source['video_path']) == source['video_sha256'], 'Selected original video changed')
    
    old = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    require(old in (None, p['cublas_workspace_config']), 'Conflicting CUDA workspace configuration')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = p['cublas_workspace_config']
    import torch
    import cv2
    import run_flicker_multimodel_observed_inference as components
    require(torch.cuda.is_available() and torch.cuda.mem_get_info()[0] >= p['minimum_free_gpu_gib']*1024**3,
            'Need20GiB free GPU; wait rather than evict other tasks')
    torch.set_num_threads(p['threads']); cv2.setNumThreads(p['opencv_threads'])
    torch.manual_seed(p['seed']); torch.cuda.manual_seed_all(p['seed'])
    torch.cuda.set_per_process_memory_fraction(p['cuda_memory_fraction'])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    def update(**values):
        tmp = out/('status.'+uuid.uuid4().hex+'.json')
        publish(tmp, dict(pid=os.getpid(), updated_at=utc(), **values)); os.replace(tmp, out/'status.json')
    results = []
    try:
        for source in sources:
            folder = out/source['source_key']
            if folder.exists(): result = verify_completed_source(folder, source, p['_protocol_sha256'])
            else:
                require(shutil.disk_usage(out).free >= (p['reserve_disk_gib']+3)*1024**3, 'Disk reserve reached')
                result = extract_source(folder, p, source, components, update)
            results.append(dict(source_key=source['source_key'], completed_sha256=sha(folder/'completed.json'),
                                original_frames=result['original_frames'], feature_shape=result['feature_shape']))
        require(len(results) == 2 and sum(x['original_frames'] for x in results) == 360, 'Incomplete pair')
        report = dict(completed=True, protocol_sha256=sha(out/'protocol.json'), sources=results,
                      source_count=2, original_frames=360, GPU_observations=True, labels_read=False,
                      training_updates=0, temporal_predictions_generated=False, fresh_holdout=False,
                      CPU_numeric_equivalence_claimed=False, automatic_training_authorized=False)
        publish(out/'report.json', report)
        publish(out/'completed.json', dict(completed=True, protocol_sha256=sha(out/'protocol.json'), report_sha256=sha(out/'report.json')))
        update(status='completed', sources=2, original_frames=360)
    except BaseException as exc:
        update(status='failed_preserved', error=str(exc)); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('prepare', 'run'), required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        require(not args.resume, 'Prepare is fresh-only'); prepare(args.output)
        print(json.dumps(dict(prepared=str(args.output), sources=2, original_frames=360, GPU_started=False)))
    else: run(args.output, args.resume)


if __name__ == '__main__': main()
