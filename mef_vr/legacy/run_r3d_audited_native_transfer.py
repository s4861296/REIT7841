
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

from prepare_r3d_audited_native_context import require_complete_audit
from r3d_native_context_transfer import (
    ARMS, SEEDS, PHASE_COUNTS, join_contexts, validate_prepared,
    observation_decode, predict_observation, require_original_replay,
    save_observation_prediction,
)
from r3d_upstream_scope_audit_contract import require, coverage
from run_cache_encoding_consistent_control import (
    BoundReader, C_TRIAL, C_AUDIT_SHA, EVIDENCE, save, sha,
)

CHECKPOINT_SHA = '5cbbd8c4cbe5efa30a29115a3b27ed2b594e8607613fe9379f14374cc3229a98'
COHORTS = {
    'train': ('train_paired', 'train_hn'),
    'development': ('validation_paired', 'development_hn'),
}
VARIANTS = tuple(seed + '__' + arm for seed in SEEDS for arm in ARMS)
NATIVE_CORE_FILES = ('r3d_native_motion_cues.py', 'r3d_native_restoration_aux.py',
                     'r3d_native_pixel_refiner.py', 'r3d_native_pixel_data.py')


def historical_context(observation, row, *, bound_observation=None):
    
    audit = row['input_audit']
    require((row['sample_id'], row['frame_count'], row['fps']) ==
            (observation['sample_id'], observation['frame_count'], observation['fps']), 'Historical source differs')
    require(audit['video_path'] == observation['video_path'], 'Historical file path differs')
    if 'source_sha256' in audit or 'source_sha256_checked' in audit:
        require(audit.get('source_sha256') == observation['video_sha256']
                and audit.get('source_sha256_checked') is True, 'Historical embedded file hash differs')
    else:
        
        
        require(bound_observation is not None, 'Legacy row requires independently bound source manifest')
        for key in ('sample_id', 'video_path', 'video_sha256', 'frame_count', 'height', 'width', 'fps'):
            require(bound_observation[key] == observation[key], 'Legacy manifest observation differs')
    require((audit['decoded_frame_count'], audit['decoded_height'], audit['decoded_width'], audit['decoded_fps']) ==
            tuple(observation[k] for k in ('frame_count', 'height', 'width', 'fps')), 'Historical decoded source differs')
    require(row['all_source_frames_covered'] is True and audit['all_frames'] is True
            and audit['native_resolution_input'] is True
            and not any(audit[k] for k in ('spatial_resize', 'temporal_sampling', 'video_reencode')), 'Historical input was altered')
    context = dict(sample_id=observation['sample_id'], frame_probabilities=list(row['frame_probabilities']),
                   coverage=coverage(observation['frame_count'])[1])
    validate_prepared(dict(observation=observation, context=context))
    return context


def python_closure(here, first):
    
    todo = [here / first]; files = {}
    while todo:
        path = todo.pop()
        if path.name in files: continue
        require(path.parent == here and path.is_file(), 'Local code dependency missing')
        files[path.name] = path
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else [])
            for name in modules:
                candidate = here / (name.split('.')[0] + '.py')
                if candidate.is_file(): todo.append(candidate)
    return files


def prepare(prepared_reports, comparison_path, comparison_sha, out):
    require(not out.exists() and out.parent.is_dir(), 'New output under existing parent required')
    require(shutil.disk_usage(out.parent).free > 40 * 1024**3, 'Keep40GiB reserve')
    require(len(prepared_reports) == 2, 'Both completed seed reports required')
    reader = BoundReader()
    comparison = reader.read(comparison_path, comparison_sha)
    require(comparison['completed'] is True and comparison['repeated_seed_runs'] == 2
            and comparison['unique_development_sources'] == 200
            and not comparison['test_used'] and not comparison['real_gt_used'], 'Completed original two-seed comparison required')
    by_seed, upstream = {}, {}
    for report_path, report_sha in prepared_reports:
        report = reader.read(report_path, report_sha)
        require(report['completed'] is True and report['unique_source_videos'] == 500
                and report['context_rows'] == 1000 and report['rows_per_arm'] == PHASE_COUNTS
                and not report['test_used'] and not report['real_gt_used'], 'Complete prepared context report required')
        audit_path = Path(report['source_audit'])
        require(comparison['source_sha256'].get(str(audit_path)) == report['source_audit_sha256'], 'Comparison must bind the same independent audit')
        audit = reader.read(audit_path, report['source_audit_sha256']); require_complete_audit(audit)
        parent = Path(report['source_run'])
        require(str(parent) == audit['source_run'], 'Source run differs')
        p = reader.read(parent / 'protocol.json', audit['parent_protocol_sha256'])
        seed = str(p['seed'])
        require(seed in SEEDS and seed not in by_seed, 'Wrong/repeated seed')
        
        for path, digest in report['source_sha256'].items(): reader.bind(path, digest)
        by_seed[seed] = {}
        for phase in PHASE_COUNTS:
            by_seed[seed][phase] = {}
            for arm in ARMS:
                name = arm + '_' + phase + '.json'
                by_seed[seed][phase][arm] = reader.read(report_path.parent / name, report['output_sha256'][name])
        upstream[seed] = dict(preparation=str(report_path), preparation_sha256=report_sha,
                              audit=str(audit_path), audit_sha256=report['source_audit_sha256'],
                              scientific_gate_passed=report['scientific_screening_passed'], models=report['models'])
    joined = join_contexts(by_seed)
    old_audit_path = C_TRIAL / 'formal_audit/audit.json'
    old_audit = reader.read(old_audit_path, C_AUDIT_SHA); old = old_audit['source_hashes']
    require(old_audit['passed'] is True and not old_audit['execution_smoke']
            and not old_audit['test_used'] and not old_audit['real_gt_used'], 'Complete original C audit required')
    cp_path = C_TRIAL / 'formal/protocol.json'
    cp = reader.read(cp_path, old[str(cp_path)])
    ep_path = EVIDENCE / 'protocol.json'
    ep = reader.read(ep_path, old[str(ep_path)])
    checkpoint = C_TRIAL / 'formal/C_zero_motion/checkpoint-0003/training_state.pt'
    reader.bind(checkpoint, CHECKPOINT_SHA)
    observations, evaluation_records = {}, {}
    legacy_hash_fields = dict(native_reference=0, historical_H=0)
    for phase, cohorts in COHORTS.items():
        ref, masks, h_rows, records = {}, {}, {}, {}
        for cohort in cohorts:
            folder = checkpoint.parent / 'evaluation' / cohort
            pp = folder / 'predictions.jsonl'
            ip = folder / 'mask_inventory.json'
            hp = EVIDENCE / (cohort + '_predictions.jsonl')
            manifest = EVIDENCE / ep['cohort_manifests'][cohort]
            for target, rows in ((ref, reader.rows(pp, old[str(pp)])),
                                 (masks, reader.read(ip, old[str(ip)])),
                                 (h_rows, reader.rows(hp, ep['output_sha256'][hp.name])),
                                 (records, reader.rows(manifest, ep['output_sha256'][manifest.name]))):
                for row in rows:
                    require(row['sample_id'] not in target, 'Duplicate original source')
                    target[row['sample_id']] = row
        ids = {r['observation']['sample_id'] for r in joined[phase]}
        require(all(set(values) == ids for values in (ref, masks, h_rows, records)), 'Original complete population mismatch')
        observations[phase] = []
        for row in joined[phase]:
            obs = row['observation']; sid = obs['sample_id']; item = masks[sid]
            require(item['video_sha256'] == obs['video_sha256'], 'Native reference source differs')
            reader.bind(item['path'], item['sha256'])
            require(old.get(item['path']) == item['sha256'], 'Native baseline mask must be audit-bound')
            
            manifest_observation = {k: records[sid][k] for k in
                ('sample_id', 'video_path', 'video_sha256', 'frame_count', 'height', 'width', 'fps')}
            historical_context(obs, ref[sid], bound_observation=manifest_observation)
            row['contexts']['C_original_H'] = historical_context(obs, h_rows[sid], bound_observation=manifest_observation)
            legacy_hash_fields['native_reference'] += 'source_sha256' not in ref[sid]['input_audit']
            legacy_hash_fields['historical_H'] += 'source_sha256' not in h_rows[sid]['input_audit']
            row['reference'] = dict(mask_path=item['path'], mask_sha256=item['sha256'],
                                   frame_probabilities=list(ref[sid]['frame_probabilities']))
            observations[phase].append(row)
        evaluation_records[phase] = [records[sid] for sid in sorted(ids)]
    for field in ('sample_id', 'scene_id', 'video_sha256'):
        require(not {r[field] for r in evaluation_records['train']}
                & {r[field] for r in evaluation_records['development']}, 'Current-stage train/Dev overlap')
    here = Path(__file__).resolve().parent
    code = python_closure(here, Path(__file__).name)
    
    code.update(python_closure(here, 'report_r3d_audited_native_transfer.py'))
    for name in NATIVE_CORE_FILES: reader.bind(code[name], cp['code_sha256'][name])
    reader.verify()
    out.mkdir(); (out / 'code').mkdir()
    for name, path in code.items():
        reader.bind(path); shutil.copy2(path, out / 'code' / name)
    for name in ('test_r3d_native_context_transfer.py', 'test_r3d_audited_native_transfer_runner.py',
                 'test_r3d_native_transfer_scoring.py', 'test_r3d_native_transfer_report.py'):
        path = here.parent / 'tests' / name; reader.bind(path); shutil.copy2(path, out / name)
    for phase, rows in observations.items(): save(out / (phase + '_observations.json'), rows)
    
    save(out / 'evaluation_records.json', evaluation_records)
    protocol = dict(protocol='audited_two_seed_frozen_C_context_transfer_v1', created_at=datetime.now(timezone.utc).isoformat(),
        upstream=upstream, original_C_checkpoint=str(checkpoint), original_C_checkpoint_sha256=CHECKPOINT_SHA,
        phase_counts=PHASE_COUNTS, variants=list(VARIANTS), original_chain_replays=500,
        new_context_native_inferences=2000, unique_source_videos=500, original_frames_per_variant=67200,
        training_updates=0, threshold=.5, core_frames=8, halo_frames=4, postprocessing='none',
        input_mode='original_mp4', precision='FP32 deterministic; no AMP or TF32',
        all_original_frames=True, original_fps=True, native_resolution=True, original_absolute_pts=True,
        gt_in_model_forward=False, paired_clean_in_model_forward=False,
        original_chain_must_finish_before_any_new_context=True,
        original_replay='exact binary/PTS; native maxima atol=rtol=1e-6',
        legacy_rows_without_embedded_video_sha256=legacy_hash_fields,
        legacy_hash_provenance='Original hash-bound source manifest, native mask inventory, source filepath and observation metadata; no fabricated embedded historical check',
        disk_reserve_gib=40, gpu_free_required_gib=24, gpu_memory_fraction=.65,
        test_used=False, real_gt_used=False, automatic_promotion=False,
        scientific_decision='Diagnostic only; no new passing gate or best-seed selection',
        post_inference_scoring_frozen_before_prediction=True,
        scoring_modules=['r3d_native_transfer_scoring.py','report_r3d_audited_native_transfer.py'],
        source_sha256=reader.bindings,
        code_sha256={name: sha(out / 'code' / name) for name in code},
        input_sha256={f.name: sha(f) for f in out.glob('*.json')},
        limitations=['Frozen C received historical H during training; replacement is a distribution-shift diagnostic, not joint retraining.',
            'Repeated synthetic Train/Dev, historical upstream HN exposure, only5 paired Dev scenes.',
            'Native pixel and joint-event scores require a separate isolated scorer and independent audit.',
            'Binary masks plus frame maxima do not retain full native probability volumes for recalibration.',
            'No Test/real evaluation or new root-cause classification; no claim of final capability.'])
    require(sum(r['observation']['frame_count'] for rows in observations.values() for r in rows) == 67200, 'Complete source frame budget differs')
    reader.verify(); save(out / 'protocol.json', protocol)
    save(out / 'preflight.json', dict(completed=True, protocol_sha256=sha(out / 'protocol.json'),
                                    model_inferences=0, training_updates=0, native_scoring_completed=False))
    return protocol


def load_reference(reader, item):
    reference = item['reference']
    reader.bind(reference['mask_path'], reference['mask_sha256'])
    with np.load(reference['mask_path'], allow_pickle=False) as data:
        result = {k: data[k] for k in ('packed_masks', 'native_shape', 'source_time_s', 'pixel_threshold')}
    result['frame_probabilities'] = np.asarray(reference['frame_probabilities'], np.float64)
    require_original_replay(result, result)
    return result


def run(out):
    import torch
    from r3d_native_motion_cues import NativeMotionCueRefiner
    from r3d_native_zero_motion_inference import predict_zero_motion_chunks
    from run_cache_encoding_consistent_control import model_hash
    require(Path(__file__).resolve().parent == out / 'code', 'Run frozen snapshot only')
    protocol = json.loads((out / 'protocol.json').read_text()); protocol_sha = sha(out / 'protocol.json')
    state = dict(pid=os.getpid(), protocol_sha256=protocol_sha)
    def update(**fields):
        state.update(fields); state['updated_at'] = datetime.now(timezone.utc).isoformat(); save(out / 'status.json', state)
    def reserve():
        require(shutil.disk_usage(out).free > 40 * 1024**3, 'Keep40GiB reserve; no deletion')
    try:
        update(status='preflight')
        require(protocol['protocol'] == 'audited_two_seed_frozen_C_context_transfer_v1'
                and protocol['phase_counts'] == PHASE_COUNTS and protocol['variants'] == list(VARIANTS), 'Frozen complete experiment required')
        require(not (out / 'original_chain').exists() and not (out / 'contexts').exists(), 'Existing run cannot be overwritten or silently resumed')
        reader = BoundReader(protocol['source_sha256']); reader.verify()
        for name, digest in protocol['code_sha256'].items(): reader.bind(out / 'code' / name, digest)
        for name, digest in protocol['input_sha256'].items(): reader.bind(out / name, digest)
        reserve()
        require(torch.cuda.is_available() and torch.cuda.mem_get_info()[0] >= 24 * 1024**3, 'Need24GiB free; never evict another task')
        torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.65)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        checkpoint = torch.load(protocol['original_C_checkpoint'], map_location='cpu', weights_only=False)
        require(checkpoint['arm'] == 'C_zero_motion' and checkpoint['step'] == 300, 'Original C epoch3 required')
        model = NativeMotionCueRefiner(False).cuda().eval().requires_grad_(False)
        model.load_state_dict(checkpoint['model'], strict=True); del checkpoint
        before = model_hash(model)
        records = {phase: reader.read(out / (phase + '_observations.json')) for phase in PHASE_COUNTS}
        require(all(len(records[p]) == PHASE_COUNTS[p] for p in records), 'Complete population required')
        def predict(rgb, scores, fps, times):
            return predict_zero_motion_chunks(model, rgb, scores, fps, times)
        inventory, replays = [], []
        for stage in ('original_chain', 'contexts'):
            if stage == 'contexts':
                require(len(replays) == 500 and all(r['passed'] for r in replays), 'Entire original chain must pass first')
                require(model_hash(model) == before, 'Original chain changed frozen C')
                save(out / 'original_replay.json', dict(completed=True, sources=500, frames=sum(r['frames'] for r in replays),
                    results=replays, model_states_unchanged=True))
            folder = out / stage; folder.mkdir()
            variants = ('C_original_H',) if stage == 'original_chain' else VARIANTS
            for variant in variants: (folder / variant).mkdir()
            source_index = 0
            for phase, rows in records.items():
                for row in rows:
                    reserve(); source_index += 1; obs = row['observation']; sid = obs['sample_id']
                    update(status='predicting_' + stage, phase=phase, source_index=source_index, source_total=500, sample_id=sid)
                    rgb, clock = observation_decode(obs)
                    for variant in variants:
                        update(variant=variant)
                        prepared = dict(observation=obs, context=row['contexts'][variant])
                        result = predict_observation(predict, rgb, clock, prepared)
                        if stage == 'original_chain':
                            ref = load_reference(reader, row)
                            replays.append(dict(sample_id=sid, **require_original_replay(result, ref))); del ref
                        path = folder / variant / (hashlib.sha256(sid.encode()).hexdigest()[:24] + '.npz')
                        item = save_observation_prediction(path, result)
                        inventory.append(dict(item, sample_id=sid, phase=phase, variant=variant,
                                              video_sha256=obs['video_sha256'], rgb_sha256=obs['rgb_sha256']))
                        del result
                    del rgb
                    save(out / 'inventory.json', inventory)
        require(len(inventory) == 2500 and len({(r['sample_id'], r['variant']) for r in inventory}) == 2500, 'Incomplete native predictions')
        require(model_hash(model) == before and all(p.grad is None for p in model.parameters()), 'Frozen C changed')
        update(status='final_integrity_check')
        reader.verify(); require(sha(out / 'protocol.json') == protocol_sha, 'Protocol changed')
        for row in inventory: require(sha(row['path']) == row['sha256'], 'Saved prediction changed')
        save(out / 'report.json', dict(completed=True, protocol_sha256=protocol_sha, source_sha256=reader.bindings,
            original_chain_replays=500, new_context_native_inferences=2000, unique_sources=500,
            original_source_frames_per_variant=67200, all_original_frames=True, original_fps=True, original_resolution=True,
            model_states_unchanged=True, model_state_sha256=before, parameter_updates=0,
            gt_mask_files_opened=0, native_metric_scoring_completed=False, independent_audit_pending=True,
            output_sha256={name: sha(out / name) for name in ('inventory.json', 'original_replay.json')},
            test_used=False, real_gt_used=False, automatic_promotion=False, limitations=protocol['limitations']))
        update(status='completed', report_sha256=sha(out / 'report.json'))
    except BaseException as error:
        update(status='failed', error=repr(error)); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', nargs=2, action='append', metavar=('REPORT', 'SHA'))
    parser.add_argument('--comparison', type=Path); parser.add_argument('--comparison-sha')
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--run', action='store_true')
    args = parser.parse_args(); out = args.output.resolve()
    if args.run: run(out); return
    require(args.prepared is not None and args.comparison is not None and args.comparison_sha is not None, 'Complete preparation arguments required')
    prepare([(Path(path).resolve(), digest) for path, digest in args.prepared], args.comparison.resolve(), args.comparison_sha, out)
    print(json.dumps(dict(prepared=True, output=str(out), training_started=False, inference_started=False)))


if __name__ == '__main__': main()
