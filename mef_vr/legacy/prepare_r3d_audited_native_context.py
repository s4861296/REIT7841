
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from run_cache_encoding_consistent_control import BoundReader, save, sha
from r3d_upstream_scope_audit_contract import coverage, require
from r3d_native_video_store import validate_source_clock

ARMS = ('A_head_only', 'B_early_r3d')
CLOCK_KEYS = ('fps_fraction', 'time_base_fraction', 'source_pts', 'source_time_s')


def require_complete_audit(audit):
    require(audit['passed'] is True and audit['completed'] is True
            and audit['gpu_replay']['completed'] is True
            and audit['gpu_replay']['source_inferences'] == 400
            and audit['gpu_replay']['unique_sources'] == 200
            and audit['gpu_replay']['parameter_updates'] == 0
            and audit['gpu_replay']['all400_replays_exact'] is True
            and audit['gpu_replay']['model_states_unchanged'] is True
            and not audit['test_used'] and not audit['real_gt_used'], 'Complete original-scope independent audit required')


def observation_context(record, prediction, decoded):
    
    n, h, w = (record[k] for k in ('frame_count', 'height', 'width'))
    require(all(type(v) is int and v > 0 for v in (n, h, w)), 'Native dimensions required')
    for key in ('sample_id', 'video_sha256', 'frame_count', 'fps'):
        require(prediction[key] == record[key], 'Context source differs: '+key)
    for key in ('video_path', 'video_sha256'):
        require(decoded[key] == record[key], 'Decoded source differs: '+key)
    require((decoded['decoded_frame_count'], decoded['decoded_height'], decoded['decoded_width']) == (n,h,w), 'Decoded shape differs')
    require(decoded['decoded_fps'] == record['fps'], 'Decoded FPS differs')
    require(decoded['all_frames'] is True and decoded['native_resolution_input'] is True
            and all(decoded[k] is False for k in ('spatial_resize','temporal_sampling','video_reencode')), 'Altered observation')
    clock = decoded['clock']
    require((clock['frame_count'],clock['height'],clock['width'],clock['fps']) == (n,h,w,record['fps']), 'Clock identity differs')
    validate_source_clock(clock['source_pts'],clock['time_base_fraction'],clock['fps_fraction'])
    times = [float(Fraction(t)*Fraction(clock['time_base_fraction'])) for t in clock['source_pts']]
    require(len(times) == n and times == clock['source_time_s'] == prediction['source_time_s'], 'Absolute source PTS differs')
    require(abs(float(Fraction(clock['fps_fraction']))-record['fps']) < 1e-4, 'Rational FPS differs')
    require(prediction['all_source_frames_covered'] is True
            and prediction['coverage'] == coverage(n)[1], 'Incomplete original window coverage')
    scores = np.asarray(prediction['frame_probabilities'],np.float64)
    require(scores.shape == (n,) and np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all(), 'Invalid complete context scores')
    observation = {k:record[k] for k in ('sample_id','video_path','video_sha256','frame_count','height','width','fps')}
    observation.update(rgb_sha256=decoded['rgb_sha256'], input_mode='original_mp4',
        clock={k:list(clock[k]) if isinstance(clock[k],list) else clock[k] for k in CLOCK_KEYS})
    context = dict(sample_id=record['sample_id'],frame_probabilities=list(prediction['frame_probabilities']),
                   coverage=list(prediction['coverage']))
    return dict(observation=observation,context=context)


def require_decoded_observation(rgb, clock, prepared):
    
    obs = prepared['observation']; n,h,w = (obs[k] for k in ('frame_count','height','width'))
    require(obs['input_mode'] == 'original_mp4', 'Do not mix encoded and PNG inputs')
    require(isinstance(rgb,np.ndarray) and rgb.dtype == np.uint8 and rgb.shape == (n,h,w,3)
            and rgb.flags.c_contiguous, 'Original complete contiguous RGB required')
    require(hashlib.sha256(memoryview(rgb)).hexdigest() == obs['rgb_sha256'], 'Native RGB and upstream RGB differ')
    require((clock['frame_count'],clock['height'],clock['width'],clock['fps']) == (n,h,w,obs['fps']), 'Native clock dimensions/FPS differ')
    require({k:clock[k] for k in CLOCK_KEYS} == obs['clock'], 'Native absolute clock differs')
    require(prepared['context']['sample_id'] == obs['sample_id'], 'Context source substitution')
    scores = np.asarray(prepared['context']['frame_probabilities'],np.float64)
    require(scores.shape == (n,) and np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all(), 'Invalid context')
    require(prepared['context']['coverage'] == coverage(n)[1], 'Context coverage differs')
    return scores, np.asarray(obs['clock']['source_time_s'],np.float64)


def prepare(audit_path, expected_audit_sha, output):
    require(not output.exists() and output.parent.is_dir(), 'New output directory required')
    require(shutil.disk_usage(output.parent).free > 40*1024**3, 'Keep40GiB reserve')
    reader = BoundReader(); audit = reader.read(audit_path,expected_audit_sha)
    require_complete_audit(audit)
    reader.read(audit_path.parent/'protocol.json',audit['protocol_sha256'])
    parent = Path(audit['source_run']); bindings = audit['source_sha256']
    def bound(path): return reader.read(path,bindings[str(path)])
    protocol = reader.read(parent/'protocol.json',audit['parent_protocol_sha256'])
    require(tuple(protocol['arms']) == ARMS and protocol['primary_checkpoint_epoch'] == 3
            and (protocol['window_frames'],protocol['window_stride']) == (32,8)
            and protocol['protocol'] == 'matched_upstream_early_scope_train100_v1'
            and protocol['all_original_frames'] is True and protocol['original_fps'] is True
            and protocol['original_absolute_pts'] is True and protocol['source_resize'] is False,
            'Unsupported source protocol')
    inventory = bound(parent/'input_inventory.json'); data = {}; models = {}; cohorts = {}
    for phase,names,count in [('train',('train_paired','train_hn'),300),('development',('validation_paired','development_hn'),200)]:
        records=[]
        for name in names:
            f=parent/'manifests'/(name+'.jsonl');records += reader.rows(f,bindings[str(f)])
        require(len(records) == len({r['sample_id'] for r in records}) == count, 'Complete unique cohort required')
        cohorts[phase]=records; by={r['sample_id']:r for r in records}
        for arm in ARMS:
            folder=parent/arm/'checkpoint-0003'; checkpoint=folder/'training_state.pt'
            reader.bind(checkpoint,bindings[str(checkpoint)])
            models[arm]=dict(checkpoint=str(checkpoint),checkpoint_sha256=bindings[str(checkpoint)],epoch=3,
                model_branch='backbone + head',upstream_threshold_applied=False)
            path=folder/'evaluation'/(phase+'_predictions.json');rows=bound(path)
            require(len(rows)==count and len({r['sample_id'] for r in rows})==count
                    and {r['sample_id'] for r in rows}==set(by), 'Wrong prediction population')
            data[arm+'_'+phase]=[observation_context(by[row['sample_id']],row,inventory[row['sample_id']]) for row in rows]
    for key in ('sample_id','scene_id','video_sha256'):
        require(not {r[key] for r in cohorts['train']} & {r[key] for r in cohorts['development']}, 'Current-stage train/development overlap')
    reader.verify(); output.mkdir(); code=output/'code';code.mkdir()
    here=Path(__file__).resolve().parent
    for name in (Path(__file__).name,'run_cache_encoding_consistent_control.py','r3d_upstream_scope_audit_contract.py','r3d_native_video_store.py'):
        reader.bind(here/name)
        shutil.copy2(here/name,code/name)
    test=here.parent/'tests/test_r3d_audited_native_context.py';reader.bind(test);shutil.copy2(test,output/test.name)
    for name,rows in data.items(): save(output/(name+'.json'),rows)
    report=dict(completed=True,purpose='audited observation-context preparation only',
        source_run=str(parent),source_audit=str(audit_path),source_audit_sha256=expected_audit_sha,
        unique_source_videos=500,context_rows=1000,rows_per_arm=dict(train=300,development=200),models=models,
        scientific_screening_passed=audit['temporal_screening']['passed'],automatic_promotion=False,
        source_observations_freshly_decoded=False,native_model_inferences=0,upstream_model_inferences=0,
        training_updates=0,gt_mask_files_opened=0,test_used=False,real_gt_used=False,
        source_sha256=reader.bindings,code_sha256={f.name:sha(f) for f in code.iterdir()},
        test_sha256=sha(output/test.name),output_sha256={name+'.json':sha(output/(name+'.json')) for name in data},
        limitations=['Only prepared inputs; native spatial inference/evaluation is NOT implemented by this tool.',
            'Downstream must decode RGB, check file hash and require_decoded_observation before model forward.',
            'No source labels, root names, paired Clean reference or GT masks are exposed as model inputs.',
            'Source IDs and paths are opaque provenance and must never become model features.',
            'Both A/B included regardless of failed scientific gate; no best-seed or arm promotion.',
            'Repeated Dev and historical upstream HN exposure remain; no real-world capability claim.'])
    require('torch' not in sys.modules,'Preparation must not load a model framework')
    reader.verify();save(output/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--audit-sha',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    report=prepare(args.audit,args.audit_sha,args.output)
    print(json.dumps({k:report[k] for k in ('completed','unique_source_videos','context_rows','scientific_screening_passed','native_model_inferences')}))


if __name__=='__main__':main()
