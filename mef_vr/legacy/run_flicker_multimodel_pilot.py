
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

PARENT = Path('/mnt/d/VRFlickerLab/model_training/r3d18_flicker_v1/dense_early_native_train100_formal_20261005_v2')
FLOW_TRAIN = Path('/mnt/d/VRFlickerLab/model_evaluation/r3d18_flicker_v1/native_motion_full_cache_after_E100_20260925_v1/train300')
FLOW_DEV = Path('/mnt/d/VRFlickerLab/model_evaluation/r3d18_flicker_v1/native_motion_full_cache_recovery_20260925_v2/development200')
ARMS = ('motion_context', 'motion_context_dino')


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def select_pilot(rows):
    
    bugs = defaultdict(list)
    for row in rows:
        if row['role'] == 'bug':
            bugs[(row['root_cause_family'], row['camera_condition'])].append(row)
    if len(bugs) != 10:
        raise ValueError('Expected five roots x two camera conditions')
    selected = []
    pairs = defaultdict(list)
    for row in rows:
        if row['role'] in ('bug', 'clean_or_fixed'):
            pairs[row['pair_id']].append(row)
    for key in sorted(bugs):
        bug = min(bugs[key], key=lambda r: hashlib.sha256(r['sample_id'].encode()).hexdigest())
        pair = pairs[bug['pair_id']]
        if Counter(r['role'] for r in pair) != {'bug': 1, 'clean_or_fixed': 1}:
            raise ValueError('Incomplete pair')
        selected.extend(sorted(pair, key=lambda r: r['role']))
    negatives = defaultdict(list)
    for row in rows:
        if row['role'] == 'hard_negative':
            negatives[row['hard_negative_category']].append(row)
    if len(negatives) != 5:
        raise ValueError('Five normal dynamic categories required')
    for key in sorted(negatives):
        selected.extend(sorted(negatives[key], key=lambda r: hashlib.sha256(r['sample_id'].encode()).hexdigest())[:2])
    if len(selected) != 30 or len({r['sample_id'] for r in selected}) != 30:
        raise ValueError('Pilot requires30 unique sources per phase')
    return selected


def key_for(sid):
    return hashlib.sha256(sid.encode()).hexdigest()[:24]


def load_inputs(out):
    p = json.loads((out / 'protocol.json').read_text())
    for name, digest in p['code_sha256'].items():
        if sha(out / 'code' / name) != digest:
            raise ValueError('Snapshot changed: ' + name)
    for name, digest in p['input_sha256'].items():
        if sha(out / name) != digest:
            raise ValueError('Frozen input changed: ' + name)
    for name, item in p['weights_provenance']['files'].items():
        if sha(Path(p['weights']) / name) != item['sha256']:
            raise ValueError('Frozen DINO weights changed')
    return p, json.loads((out / 'inputs.json').read_text())


def prepare(out, weights, full=False):
    from run_r3d_audited_native_transfer import python_closure
    if out.exists():
        raise FileExistsError('Use a fresh experiment; never overwrite old results')
    records = json.loads((PARENT / 'records.json').read_text())
    observations = json.loads((PARENT / 'observations.json').read_text())
    obs = {r['observation']['sample_id']: r for rows in observations.values() for r in rows}
    chosen = {phase: (rows if full else select_pilot(rows)) for phase, rows in records.items()}
    flow, provenance = {}, {}
    for phase, directory in [('train', FLOW_TRAIN), ('development', FLOW_DEV)]:
        inventory = json.loads((directory / 'inventory.json').read_text())
        flow[phase] = {row['sample_id']: row for row in inventory}
        provenance[phase] = json.loads((directory / 'protocol.json').read_text())['producer']
    inputs = {}
    for phase, rows in chosen.items():
        inputs[phase] = []
        for record in rows:
            sid = record['sample_id']
            observed = obs[sid]
            item = flow[phase][sid]
            if record['video_sha256'] != item['source']['source_video_sha256']:
                raise ValueError('Flow belongs to a different source')
            if observed['observation']['clock']['source_pts'] != item['source']['pts']:
                raise ValueError('Flow PTS differ from RGB')
            inputs[phase].append({'record': record, 'observed': observed,
                                  'flow': item, 'flow_producer': provenance[phase]})
    for field in ('sample_id', 'video_sha256', 'scene_id', 'pair_id'):
        left = {i['record'][field] for i in inputs['train']}
        right = {i['record'][field] for i in inputs['development']}
        if left & right:
            raise ValueError('Current-stage Train/Dev overlap: ' + field)
    out.mkdir(parents=True)
    if shutil.disk_usage(out).free < 60 * 1024**3:
        raise OSError('Keep60GiB free; old artifacts left untouched')
    (out / 'code').mkdir()
    here = Path(__file__).resolve().parent
    code = python_closure(here, Path(__file__).name)
    code.update(python_closure(here, 'flicker_multimodel_core.py'))
    
    for name in ('native_motion_sequence_cache.py', 'r3d_native_context_transfer.py',
                 'r3d_native_transfer_scoring.py', 'report_native_tolerant_events.py'):
        code.update(python_closure(here, name))
    for name, source in code.items():
        shutil.copy2(source, out / 'code' / name)
    test = here.parent / 'tests/test_flicker_multimodel_pilot.py'
    shutil.copy2(test, out / test.name)
    save(out / 'inputs.json', inputs)
    weights_provenance = json.loads((weights / 'provenance.json').read_text())
    p = {'version': 'multimodel_dense_state_v1', 'prepared_at': utc(),
         'scope': 'full_train300_dev200' if full else 'pilot_train30_dev30',
         'seed': 20261007, 'arms': ARMS, 'epochs': 6, 'learning_rate': .001,
         'weight_decay': .0001, 'gradient_clip': 1., 'threshold': .5,
         'loss': 'balanced state BCE +0.2 boundary BCE +0.2 auxiliary patch-presence BCE',
         'whole_video_update': True, 'triplet_update': True, 'native_input': True,
         'all_original_frames': True, 'original_fps_pts': True, 'source_resize': False,
         'dino_padding_only_patch_multiple': 14, 'dino_frozen': True, 'dino_precision': 'BF16 inference; FP16 feature cache',
         'temporal_stride': 1, 'temporal_reference_lags': [1, 4, 8], 'offline_future_context': True,
         'nominal_parameter_match': True, 'ablation': 'DINO-specific feature channels zero in motion_context',
         'effective_input_rank_match': False, 'pixel_gt_enters_forward': False,
         'paired_clean_enters_forward': False, 'segmentation_model_used': False,
         'spatial_output': 'old frozen C native mask gated by new frame decision; no new precise pixel decoder',
         'auxiliary_grid': '14x14 patch presence, not precise abnormal-pixel output',
         'automatic_promotion': False, 'test_used': False, 'real_gt_used': False,
         'validation': 'fixed final epoch6; no threshold tuning or intermediate Dev selection',
         'limitations': ['Repeated synthetic Dev; upstream R3D previously exposed to HN Dev scenes.',
                        'Pilot is a feasibility experiment, not an independent real-world result.',
                        'New temporal decisions can gate but not repair old spatial mask boundaries.',
                        'Observed RAFT cache has residual/support/consistency, not raw trajectories.',
                        'DINOv3 public download returned401; this experiment uses DINOv2-small.'],
         'weights': str(weights), 'weights_provenance': weights_provenance,
         'counts': {k: len(v) for k, v in inputs.items()},
         'frames': {k: sum(i['record']['frame_count'] for i in v) for k, v in inputs.items()},
         'source_manifest_sha256': {str(PARENT / name): sha(PARENT / name) for name in ('records.json', 'observations.json')},
         'code_sha256': {name: sha(out / 'code' / name) for name in code},
         'input_sha256': {'inputs.json': sha(out / 'inputs.json')},
         'disk_reserve_gib': 40, 'cache_limit_gib': 120}
    save(out / 'protocol.json', p)
    environment = dict(os.environ, PYTHONPATH=str(out / 'code'), PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    check = subprocess.run([sys.executable, '-m', 'unittest', 'test_flicker_multimodel_pilot', '-v'],
                           cwd=out, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    save(out / 'tests.json', {'returncode': check.returncode, 'output': check.stdout})
    if check.returncode:
        raise RuntimeError(check.stdout)
    print(json.dumps({'prepared': True, 'output': str(out), 'counts': p['counts'], 'frames': p['frames']}), flush=True)


def source_masks(item):
    reference = item['observed']['reference']
    if sha(reference['mask_path']) != reference['mask_sha256']:
        raise ValueError('Original model prediction changed')
    r = item['record']
    with np.load(reference['mask_path'], allow_pickle=False) as z:
        packed = z['packed_masks']
        shape = tuple(int(x) for x in z['native_shape'])
        if shape != (r['frame_count'], r['height'], r['width']):
            raise ValueError('Original model mask coordinates differ')
        masks = np.unpackbits(packed, axis=1, count=r['height'] * r['width'], bitorder='little')
    return masks.reshape(shape).astype(bool)


def extract(out, p, inputs, update):
    import torch
    from transformers import Dinov2Model
    from flicker_multimodel_core import dino_embeddings, make_observation_features
    from native_motion_sequence_cache import MotionSequenceReader
    from r3d_native_context_transfer import observation_decode
    cache = out / 'features'
    cache.mkdir(exist_ok=True)
    model = None
    complete, started = 0, time.monotonic()
    for phase, rows in inputs.items():
        for item in rows:
            r = item['record']; sid = r['sample_id']; stem = key_for(sid)
            path, metadata = cache / (stem + '.npy'), cache / (stem + '.json')
            if metadata.exists():
                previous = json.loads(metadata.read_text())
                if previous['video_sha256'] != r['video_sha256'] or sha(path) != previous['features_sha256']:
                    raise ValueError('Existing feature cache changed')
                complete += 1
                continue
            if path.exists():
                raise FileExistsError('Uncommitted feature cache retained; explicit recovery needed')
            if shutil.disk_usage(out).free < p['disk_reserve_gib'] * 1024**3:
                raise OSError('Disk reserve reached')
            used = sum(f.stat().st_size for f in cache.glob('*.npy'))
            if used > p['cache_limit_gib'] * 1024**3:
                raise OSError('Derived feature cache quota reached')
            if model is None:
                model = Dinov2Model.from_pretrained(p['weights'], local_files_only=True,
                    attn_implementation='sdpa').eval().cuda().requires_grad_(False)
            update(status='extracting_native_dino', phase=phase, sample_id=sid,
                   completed_sources=complete, total_sources=sum(p['counts'].values()), frame=0)
            rgb, clock = observation_decode(item['observed']['observation'])
            began = time.monotonic()
            embedding = dino_embeddings(model, rgb, progress=lambda f, n: update(frame=f, source_frames=n))
            update(status='assembling_motion_appearance_evidence', frame=0)
            flow = item['flow']
            if sha(flow['path']) != flow['cache_sha256']:
                raise ValueError('Frozen motion cache changed')
            baseline = source_masks(item)
            with MotionSequenceReader(flow['path'], flow['source'], item['flow_producer']) as reader:
                feature, names = make_observation_features(rgb, embedding,
                    item['observed']['context']['frame_probabilities'], baseline, reader,
                    progress=lambda f, n: update(frame=f, source_frames=n))
            temporary = path.with_suffix('.partial')
            if temporary.exists():
                raise FileExistsError('Partial feature preserved; explicit recovery needed')
            with temporary.open('xb') as stream:
                np.save(stream, feature, allow_pickle=False)
            temporary.rename(path)
            meta = {'sample_id': sid, 'video_sha256': r['video_sha256'], 'shape': list(feature.shape),
                    'features_sha256': sha(path), 'feature_names': names, 'all_original_frames': True,
                    'source_time_s': clock['source_time_s'], 'fps': r['fps'],
                    'flow_cache_sha256': flow['cache_sha256'], 'dino_revision': p['weights_provenance']['revision'],
                    'labels_read': False, 'paired_clean_used': False,
                    'seconds': time.monotonic() - began}
            save(metadata, meta)
            complete += 1
            print(json.dumps({'features': complete, 'phase': phase, 'seconds': round(meta['seconds'], 2),
                              'shape': meta['shape'], 'sample_id': sid}), flush=True)
            del rgb, embedding, baseline, feature
    del model
    torch.cuda.empty_cache()
    save(out / 'feature_completion.json', {'complete': True, 'sources': complete,
        'elapsed_seconds_this_attempt': time.monotonic() - started})


def targets(item):
    
    import torch
    from flicker_multimodel_core import pool_native
    from r3d_native_transfer_scoring import load_target
    from run_cache_encoding_consistent_control import BoundReader
    r = item['record']
    packed, fidelity = load_target(r, BoundReader())
    frame = packed.any(axis=1).astype(np.float32)
    grid = []
    for bits in packed:
        mask = np.unpackbits(bits, count=r['height'] * r['width'], bitorder='little').reshape(r['height'], r['width'])
        grid.append(pool_native(torch.from_numpy(mask.copy())[None, None].float(), kind='max')[0, 0].numpy())
    return {'frame': frame, 'grid': np.stack(grid), 'fidelity': fidelity}


def read_feature(out, item):
    stem = key_for(item['record']['sample_id'])
    return np.load(out / 'features' / (stem + '.npy'), mmap_mode='r', allow_pickle=False)


def train(out, p, inputs, update):
    import torch
    from flicker_multimodel_core import DenseStateHead, supervision_loss
    if (out / 'training_complete.json').exists():
        return
    first = inputs['train'][0]
    meta = json.loads((out / 'features' / (key_for(first['record']['sample_id']) + '.json')).read_text())
    channels = meta['shape'][1]
    dino_indices = [i for i, name in enumerate(meta['feature_names']) if name.startswith('dino_')]
    
    normalization_path = out / 'normalization.npz'
    if not normalization_path.exists():
        sums = np.zeros(channels, np.float64); squares = sums.copy(); count = 0
        for item in inputs['train']:
            z = read_feature(out, item)
            for begin in range(0, len(z), 16):
                block = np.asarray(z[begin:begin + 16], dtype=np.float32)
                sums += block.sum(axis=(0, 2, 3), dtype=np.float64)
                squares += np.square(block).sum(axis=(0, 2, 3), dtype=np.float64)
                count += block.shape[0] * block.shape[2] * block.shape[3]
        mu = (sums / count).astype(np.float32)
        sd = np.sqrt(np.maximum(squares / count - mu.astype(np.float64)**2, 1e-4)).astype(np.float32)
        with normalization_path.open('xb') as stream:
            np.savez(stream, mean=mu, std=sd, count=np.asarray(count))
    with np.load(normalization_path) as z:
        mu, sd = (torch.from_numpy(z[k].copy()).cuda()[None, :, None, None] for k in ('mean', 'std'))
    torch.manual_seed(p['seed']); np.random.seed(p['seed'])
    models, optimizers = {}, {}
    for arm in ARMS:
        torch.manual_seed(p['seed'])
        model = DenseStateHead(channels).cuda()
        models[arm] = model
        optimizers[arm] = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    pairs = defaultdict(list); negatives = []
    for item in inputs['train']:
        r = item['record']
        if r['role'] == 'hard_negative': negatives.append(item)
        else: pairs[r['pair_id']].append(item)
    pair_ids = sorted(pairs)
    negatives.sort(key=lambda i: i['record']['sample_id'])
    if len(pair_ids) != len(negatives):
        raise ValueError('One HN per complete pair required')
    triples = {pid: sorted(pairs[pid], key=lambda i: i['record']['role']) + [negatives[j]] for j, pid in enumerate(pair_ids)}
    rng = np.random.default_rng(p['seed'])
    orders = [list(rng.permutation(pair_ids)) for _ in range(p['epochs'])]
    sequence = [(epoch + 1, pid) for epoch, order in enumerate(orders) for pid in order]
    history = []
    latest = out / 'latest_checkpoint.pt'
    protocol_hash = sha(out / 'protocol.json')
    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=False)
        if state['protocol_sha256'] != protocol_hash:
            raise ValueError('Resume protocol changed')
        for arm in ARMS:
            models[arm].load_state_dict(state['models'][arm])
            optimizers[arm].load_state_dict(state['optimizers'][arm])
        history = state['history']
        if [(r['epoch'], r['pair_id']) for r in history] != sequence[:len(history)]:
            raise ValueError('Resume data order changed')
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
    target_cache = {}
    started = time.monotonic()
    for step in range(len(history), len(sequence)):
        epoch, pid = sequence[step]
        update(status='training_state_boundary_heads', epoch=epoch, step=step,
               total_steps=len(sequence), pair_id=pid)
        for optimizer in optimizers.values(): optimizer.zero_grad(set_to_none=True)
        details = []
        for item in triples[pid]:
            sid = item['record']['sample_id']
            if sid not in target_cache: target_cache[sid] = targets(item)
            target = target_cache[sid]
            frame = torch.from_numpy(target['frame']).cuda()
            grid = torch.from_numpy(target['grid']).cuda()
            feature = torch.from_numpy(np.array(read_feature(out, item), copy=True)).cuda().float()
            feature = (feature - mu) / sd
            row = {'sample_id': sid, 'frames': len(frame), 'arms': {}}
            for arm, model in models.items():
                z = feature.clone() if arm == ARMS[0] else feature
                if arm == ARMS[0]: z[:, dino_indices] = 0
                prediction = model(z)
                loss, parts = supervision_loss(prediction, frame, grid)
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                (loss / 3).backward()
                row['arms'][arm] = {'loss': float(loss.detach()), **parts}
                del prediction, loss, z
            details.append(row)
            del feature, frame, grid
        for arm, model in models.items():
            grads = [v.grad for v in model.parameters() if v.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise ValueError('Missing/nonfinite gradients')
            torch.nn.utils.clip_grad_norm_(model.parameters(), p['gradient_clip'])
            optimizers[arm].step()
        history.append({'step': step + 1, 'epoch': epoch, 'pair_id': pid, 'sources': details})
        state = {'models': {a: m.state_dict() for a, m in models.items()},
                 'optimizers': {a: o.state_dict() for a, o in optimizers.items()},
                 'history': history, 'protocol_sha256': protocol_hash,
                 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                 'channels': channels, 'dino_indices': dino_indices}
        temporary = out / 'latest_checkpoint.tmp'
        torch.save(state, temporary); temporary.replace(latest)
        save(out / 'history.json', history)
        if (step + 1) % len(pair_ids) == 0:
            checkpoint = out / f'checkpoint_epoch{epoch:02d}.pt'
            if checkpoint.exists(): raise FileExistsError('Preserve existing epoch checkpoint')
            shutil.copy2(latest, checkpoint)
        print(json.dumps({'trained_step': step + 1, 'total_steps': len(sequence), 'epoch': epoch}), flush=True)
    save(out / 'training_complete.json', {'completed': True, 'steps_per_arm': len(sequence),
        'epochs': p['epochs'], 'seconds_this_attempt': time.monotonic() - started,
        'checkpoint': str(latest), 'checkpoint_sha256': sha(latest), 'protocol_sha256': protocol_hash})
    del models, optimizers
    torch.cuda.empty_cache()


def metrics(rows):
    from report_native_tolerant_events import evaluate
    y = np.concatenate([np.asarray(r['frame_targets'], bool) for r in rows])
    pred = np.concatenate([np.asarray(r['frame_probabilities']) >= .5 for r in rows])
    tp, fp, fn, tn = (int((a & b).sum()) for a, b in ((y, pred), (~y, pred), (y, ~pred), (~y, ~pred)))
    vy = np.asarray([bool(any(r['frame_targets'])) for r in rows])
    vp = np.asarray([max(r['frame_probabilities']) >= .5 for r in rows])
    vt, vf, vn, vtn = (int((a & b).sum()) for a, b in ((vy, vp), (~vy, vp), (vy, ~vp), (~vy, ~vp)))
    def divide(a, b): return a / b if b else None
    event = evaluate(rows, 3)
    normal = {}
    for role in ('clean_or_fixed', 'hard_negative'):
        group = [r for r in rows if r['role'] == role]
        normal[role] = {'videos': len(group), 'false_positive_videos': sum(max(r['frame_probabilities']) >= .5 for r in group)}
    return {'videos': len(rows), 'frames': len(y),
            'video': {'tp': vt, 'fp': vf, 'fn': vn, 'tn': vtn, 'accuracy': float((vy == vp).mean()),
                      'f1': divide(2 * vt, 2 * vt + vf + vn)},
            'frame': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                      'f1': divide(2 * tp, 2 * tp + fp + fn), 'recall': divide(tp, tp + fn)},
            'event_tolerance3': {k: v for k, v in event.items() if k != 'records'}, 'negatives': normal}


def evaluate(out, p, inputs, update):
    import torch
    from flicker_multimodel_core import DenseStateHead
    if (out / 'results.json').exists(): return
    checkpoint = torch.load(out / 'latest_checkpoint.pt', map_location='cpu', weights_only=False)
    if checkpoint['history'][-1]['epoch'] != p['epochs']:
        raise ValueError('Final prespecified epoch required')
    models = {}
    for arm in ARMS:
        models[arm] = DenseStateHead(checkpoint['channels']).cuda().eval()
        models[arm].load_state_dict(checkpoint['models'][arm])
    with np.load(out / 'normalization.npz') as z:
        mu, sd = (torch.from_numpy(z[k].copy()).cuda()[None, :, None, None] for k in ('mean', 'std'))
    rows = {phase: {arm: [] for arm in (*ARMS, 'frozen_C')} for phase in inputs}
    predictions_dir = out / 'predictions'; predictions_dir.mkdir(exist_ok=True)
    with torch.inference_mode():
        for phase, sources in inputs.items():
            for index, item in enumerate(sources):
                r = item['record']; sid = r['sample_id']
                update(status='evaluating_final_heads', phase=phase, sample_id=sid, source_index=index,
                       total_sources=len(sources))
                features = torch.from_numpy(np.array(read_feature(out, item), copy=True)).cuda().float()
                features = (features - mu) / sd
                prediction = {}
                for arm, model in models.items():
                    z = features.clone() if arm == ARMS[0] else features
                    if arm == ARMS[0]: z[:, checkpoint['dino_indices']] = 0
                    outputs = model(z)
                    prediction[arm] = {'frame_probabilities': outputs['frame_logits'].sigmoid().cpu().tolist(),
                                      'boundary_probabilities': outputs['boundary_logits'].sigmoid().cpu().tolist()}
                
                dest = predictions_dir / (key_for(sid) + '.json')
                payload = {'sample_id': sid, 'phase': phase, 'source_time_s': item['observed']['observation']['clock']['source_time_s'],
                           'video_sha256': r['video_sha256'], 'predictions': prediction}
                if dest.exists():
                    if json.loads(dest.read_text()) != payload: raise ValueError('Replay predictions differ')
                else: save(dest, payload)
                y = targets(item)['frame'].astype(int).tolist()
                common = {k: r[k] for k in ('sample_id', 'frame_count', 'fps', 'role', 'root_cause_family', 'camera_condition')}
                common.update(frame_targets=y, all_source_frames_covered=True)
                for arm in ARMS: rows[phase][arm].append({**common, 'frame_probabilities': prediction[arm]['frame_probabilities']})
                rows[phase]['frozen_C'].append({**common, 'frame_probabilities': item['observed']['reference']['frame_probabilities']})
                del features, z, outputs
    result = {'completed': True, 'scope': p['scope'], 'test_used': False, 'real_gt_used': False,
              'automatic_promotion': False, 'new_precise_spatial_decoder': False,
              'metrics': {phase: {arm: metrics(items) for arm, items in arms.items()} for phase, arms in rows.items()},
              'protocol_sha256': sha(out / 'protocol.json'), 'checkpoint_sha256': sha(out / 'latest_checkpoint.pt'),
              'limitations': p['limitations']}
    save(out / 'scored_predictions.json', rows)
    save(out / 'results.json', result)
    print(json.dumps({'completed_evaluation': True, 'development': result['metrics']['development']}, ensure_ascii=False), flush=True)


def reviews(out, p, inputs, update):
    
    import av
    import cv2
    from report_native_tolerant_events import intervals
    directory = out / 'reviews'; directory.mkdir(exist_ok=True)
    index = []
    for number, item in enumerate(inputs['development'], 1):
        r = item['record']; sid = r['sample_id']; stem = key_for(sid)
        prediction = json.loads((out / 'predictions' / (stem + '.json')).read_text())
        probs = prediction['predictions'][ARMS[1]]['frame_probabilities']
        output = directory / f'{number:02d}_{r["role"]}_{stem}.mp4'
        row = {'video': output.name, 'sample_id': sid, 'source_video': r['video_path'],
               'predicted_intervals_0based_inclusive': intervals([x >= .5 for x in probs]),
               'warning': 'Red: frozen C spatial candidate gated by NEW temporal state. No new pixel decoder; no GT overlay.'}
        if output.exists():
            with av.open(str(output)) as check:
                if sum(1 for _ in check.decode(video=0)) != r['frame_count']:
                    raise ValueError('Partial review retained; explicit recovery required')
            index.append(row); continue
        update(status='rendering_original_speed_reviews', review=number, total_reviews=len(inputs['development']), sample_id=sid)
        masks = source_masks(item)
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{r["width"]}x{r["height"]}', '-r', str(r['fps']), '-i', '-', '-an',
               '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        seen = 0
        try:
            with av.open(r['video_path']) as source:
                for t, frame in enumerate(source.decode(video=0)):
                    rgb = frame.to_ndarray(format='rgb24')
                    if probs[t] >= .5:
                        m = masks[t]
                        rgb[m] = (rgb[m].astype(np.float32) * .55 + np.array([255, 0, 0]) * .45).astype(np.uint8)
                    cv2.rectangle(rgb, (0, 0), (r['width'], 74), (0, 0, 0), -1)
                    cv2.putText(rgb, f'Frame {t}  t={t/r["fps"]:.3f}s  NEW state p={probs[t]:.3f}',
                                (12, 27), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(rgb, 'Red = OLD spatial mask gated by NEW state; no GT overlay',
                                (12, 56), cv2.FONT_HERSHEY_SIMPLEX, .57, (255, 255, 255), 1, cv2.LINE_AA)
                    process.stdin.write(rgb.tobytes()); seen += 1
        finally:
            process.stdin.close()
        if process.wait() != 0 or seen != r['frame_count']:
            raise RuntimeError('Review encoding incomplete')
        row['sha256'] = sha(output); index.append(row)
        save(directory / 'index.json', index)
    import html
    cards = ''.join(f'<article><h3>{html.escape(i["sample_id"])}</h3><video controls preload="none" src="{i["video"]}"></video></article>' for i in index)
    (directory / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>多模型时间 Pilot</title>'
        '<style>body{font-family:system-ui;max-width:1400px;margin:24px auto}video{width:100%}article{margin:30px 0}h3{overflow-wrap:anywhere}</style>'
        '<h1>多模型逐帧状态 Pilot：原速 Review</h1><p>红色是旧 C 空间预测经新时间判断筛选，不是新的精确像素解码结果。没有叠加 GT。此处包含全部预先选定开发视频，不挑成功案例。</p>' + cards)
    save(directory / 'index.json', index)


def run(out, action):
    import torch
    import cv2
    if Path(__file__).resolve().parent != out.resolve() / 'code':
        raise ValueError('Run the frozen snapshot in output/code')
    p, inputs = load_inputs(out)
    if json.loads((out / 'tests.json').read_text())['returncode'] != 0:
        raise ValueError('Tests failed')
    lock = (out / 'worker.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'pid': os.getpid(), 'started_utc': utc(), 'action': action, 'status': 'starting'}
    def update(**values):
        state.update(values); state['updated_at'] = utc(); save(out / 'status.json', state)
    update()
    try:
        torch.set_num_threads(2); cv2.setNumThreads(1)
        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 10 * 1024**3:
            raise RuntimeError('Need10GiB free GPU; do not evict user tasks')
        torch.cuda.set_per_process_memory_fraction(.65)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if action in ('pipeline', 'extract'): extract(out, p, inputs, update)
        if action in ('pipeline', 'train'):
            if not (out / 'feature_completion.json').exists(): raise ValueError('Complete cache required before training')
            train(out, p, inputs, update)
            evaluate(out, p, inputs, update)
        if action in ('pipeline', 'review'): reviews(out, p, inputs, update)
        update(status='completed', completed_utc=utc())
    except BaseException as exc:
        update(status='failed', error=str(exc), traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('prepare', 'pipeline', 'extract', 'train', 'review'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        if args.weights is None: parser.error('--weights required')
        prepare(args.output, args.weights, args.full)
    else:
        run(args.output, args.action)


if __name__ == '__main__':
    main()
