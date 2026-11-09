
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from report_native_tolerant_events import evaluate, intervals, match_events


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values)/len(values) if values else None


def metric_token(sid, cache=None):
    
    return cache[sid]['sha256'][:16] if cache is not None else hashlib.sha256(sid.encode()).hexdigest()[:16]


SPATIAL_STATES=('missed_empty','off_target','partial_or_overwide','strict_precision90_recall90')


def spatial_state(row):
    
    c=row['pixels'];tp,fp,fn=(c[k] for k in ('tp','fp','fn'))
    if min(tp,fp,fn)<0 or tp+fn!=row['target_positive_pixels'] or tp+fp!=row['predicted_positive_pixels']:
        raise ValueError('Invalid native spatial counts')
    if tp+fn==0:return None
    if tp+fp==0:return 'missed_empty'
    if tp==0:return 'off_target'
    if 10*tp>=9*(tp+fp) and 10*tp>=9*(tp+fn):return 'strict_precision90_recall90'
    return 'partial_or_overwide'


def spatial_quality(rows):
    positives=[r for r in rows if r['target_positive_pixels']>0]
    grouped=defaultdict(list)
    states={name:0 for name in SPATIAL_STATES}
    for r in positives:
        value=spatial_state(r);states[value]+=1;grouped[r['sample_id']].append(value)
    return dict(positive_frames=len(positives),positive_cases=len(grouped),frame_counts=states,
        equal_case_mean_frame_fraction={name:mean([sum(v==name for v in values)/len(values)
            for values in grouped.values()]) for name in SPATIAL_STATES},
        convention='Strict native pixels: both P>=0.9 and R>=0.9; no dilation, not bbox/instance accuracy or a new gate')


def pixel_stratum(cases):
    
    positives = [r for r in cases if r['target_positive_pixels'] > 0]
    grouped = defaultdict(list)
    for r in positives:
        grouped[r['sample_id']].append(r)
    counts = {k: sum(r['pixels'][k] for r in cases) for k in ('tp', 'fp', 'fn', 'tn')}
    tp, fp, fn = (counts[k] for k in ('tp', 'fp', 'fn'))
    normal = [r for r in cases if r['target_positive_pixels'] == 0]
    return dict(frames=len(cases), videos=len({r['sample_id'] for r in cases}),
        positive_frames=len(positives), positive_videos=len(grouped), counts=counts,
        mean_case_positive_frame_iou=mean([mean([r['pixels']['iou'] for r in rows]) for rows in grouped.values()]),
        mean_case_positive_frame_boundary2_f1=mean([mean([r['boundary_2px']['f1'] for r in rows]) for rows in grouped.values()]),
        positive_frames_with_any_prediction=sum(r['predicted_positive_pixels'] > 0 for r in positives),
        positive_frames_with_GT_intersection=sum(r['pixels']['tp'] > 0 for r in positives),
        micro_pixel_iou=tp/(tp+fp+fn) if tp+fp+fn else None,
        micro_pixel_precision=tp/(tp+fp) if tp+fp else None,
        micro_pixel_recall=tp/(tp+fn) if tp+fn else None,
        normal_frames=len(normal), normal_frames_with_false_pixels=sum(r['predicted_positive_pixels'] > 0 for r in normal),
        normal_false_pixels=sum(r['predicted_positive_pixels'] for r in normal),
        strict_spatial_quality=spatial_quality(cases))


def temporal_strata(rows):
    metrics = {str(t): evaluate(rows, t) for t in (0, 3)}
    durations = {name: dict(gt_events=0, matched_strict=0, matched_pm3=0)
                 for name in ('1', '2_to_3', '4_to_10', 'gt_10')}
    normal_frames = false_frames = gap_frames = false_gap_frames = 0
    gap_bins = {name: dict(spans=0, frames=0, false_frames=0, all_gap_frames_false_spans=0)
                for name in ('1', '2_to_3', '4_to_10', 'gt_10')}
    for r in rows:
        y = r['frame_targets']; pred = [s >= .5 for s in r['frame_probabilities']]
        gt = intervals(y); events = intervals(pred)
        matched = {t: {i for i, _ in match_events(gt, events, t)} for t in (0, 3)}
        for i, (start, end) in enumerate(gt):
            n = end-start+1
            key = '1' if n == 1 else '2_to_3' if n <= 3 else '4_to_10' if n <= 10 else 'gt_10'
            d = durations[key]; d['gt_events'] += 1
            d['matched_strict'] += i in matched[0]; d['matched_pm3'] += i in matched[3]
        normal_frames += sum(not b for b in y)
        false_frames += sum(p and not b for p, b in zip(pred, y))
        
        for left, right in zip(gt, gt[1:]):
            a, b = left[1]+1, right[0]
            gap_frames += b-a; false_gap_frames += sum(pred[a:b])
            n = b-a
            key = '1' if n == 1 else '2_to_3' if n <= 3 else '4_to_10' if n <= 10 else 'gt_10'
            d = gap_bins[key]; d['spans'] += 1; d['frames'] += n
            d['false_frames'] += sum(pred[a:b]); d['all_gap_frames_false_spans'] += all(pred[a:b])
    for d in durations.values():
        d['recall_pm3'] = d['matched_pm3']/d['gt_events'] if d['gt_events'] else None
    return dict(videos=len(rows), endpoint_metrics={t:{k:v for k,v in m.items() if k!='records'} for t,m in metrics.items()},
        duration_bins=durations, normal_frames=normal_frames, normal_false_frames=false_frames,
        inter_event_normal_frames=gap_frames, inter_event_normal_false_frames=false_gap_frames,
        inter_event_false_frame_rate=false_gap_frames/gap_frames if gap_frames else None,
        inter_event_gap_duration_bins=gap_bins)


def validate_frames(frames, prediction):
    n = prediction['frame_count']
    if len(frames) != n or [r['frame_index'] for r in frames] != list(range(n)):
        raise ValueError('Incomplete native frame metrics')
    if n != len(prediction['frame_targets']) or n != len(prediction['frame_probabilities']):
        raise ValueError('Incomplete prediction frames')
    for r, y, score in zip(frames, prediction['frame_targets'], prediction['frame_probabilities']):
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid frame probability')
        if (r['target_positive_pixels'] > 0) != bool(y) or (r['predicted_positive_pixels'] > 0) != (score >= .5):
            raise ValueError('Pixel/time source disagreement')
        c = r['pixels']
        if min(c[k] for k in ('tp','fp','fn','tn')) < 0 or c['tp']+c['fn'] != r['target_positive_pixels'] or c['tp']+c['fp'] != r['predicted_positive_pixels']:
            raise ValueError('Invalid pixel counts')


def prediction_from_native_counts(source, frames):
    
    row = dict(sample_id=source['sample_id'], frame_count=source['frame_count'], fps=source['fps'],
        frame_targets=[int(f['target_positive_pixels'] > 0) for f in frames],
        frame_probabilities=[float(f['predicted_positive_pixels'] > 0) for f in frames])
    validate_frames(frames, row)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'audit', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--launch', action='store_true'); p.add_argument('--worker', action='store_true')
    a = p.parse_args(); out = a.output
    if not a.worker:
        out.mkdir(parents=True, exist_ok=False); code=out/'code'; code.mkdir()
        for name in (Path(__file__).name, 'report_native_tolerant_events.py'):
            shutil.copy2(Path(__file__).with_name(name), code/name)
        command=[sys.executable, str(code/Path(__file__).name), *[v for v in sys.argv[1:] if v!='--launch'], '--worker']
        if a.launch:
            with (out/'worker.log').open('x') as log:
                proc=subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (out/'launch.json').write_text(json.dumps(dict(pid=proc.pid, command=command), indent=2))
            print(json.dumps(dict(pid=proc.pid, output=str(out)))); return
        os.execv(sys.executable, command)
    state=dict(pid=os.getpid()); bindings={}
    def save(name, value):
        path=out/name; temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temp.replace(path)
    def status(**kw):
        state.update(kw); state['updated_unix_s']=time.time(); save('status.json',state)
    def raw(path):
        path=Path(path); data=path.read_bytes(); digest=hashlib.sha256(data).hexdigest()
        if str(path) in bindings and bindings[str(path)]!=digest: raise ValueError('Source changed')
        bindings[str(path)]=digest; return data
    def read(path): return json.loads(raw(path))
    try:
        while True:
            s=json.loads((a.audit/'status.json').read_text())
            if s['status']=='completed': break
            if s['status']=='failed': raise RuntimeError('Native audit failed; preserve failure')
            proc=Path('/proc')/str(s['pid'])/'cmdline'
            cmd=proc.read_bytes().decode().replace('\0',' ') if proc.exists() else ''
            if str(a.audit) not in cmd:
                if json.loads((a.audit/'status.json').read_text())['status']=='completed': break
                raise RuntimeError('Audit missing; do not wait on stale status')
            status(status='waiting_for_verified_native_audit', producer_pid=s['pid']); time.sleep(15)
        ar=read(a.audit/'report.json'); proto=read(a.run/'protocol.json')
        if not ar.get('completed') or not (ar.get('passed') or ar.get('passed_native_reconstruction')) or ar.get('execution_smoke') or ar['test_used'] or ar['real_gt_used']:
            raise ValueError('Require complete full Dev native audit')
        local_confidence = proto.get('protocol') == 'native_local_confidence_control_v1'
        continuous = local_confidence or proto.get('protocol') == 'native_continuous_prior_control_v1'
        sources = {}; producer = None
        if continuous:
            producer = read(a.run/'report.json')
            flag = 'local_confidence_control' if local_confidence else 'continuous_prior_control'
            if (ar.get('source_run') != str(a.run) or not ar.get('passed') or not ar.get(flag)
                    or ar.get('training_steps_checked') != 300 or ar.get('independent_native_video_replays') != 200
                    or ar.get('fresh_frozen_C_replays') != 200 or ar.get('independent_directional_intersections') != 200
                    or not producer['completed'] or producer['execution_smoke'] or producer['test_used'] or producer['real_gt_used']
                    or proto['test_used'] or proto['real_gt_used'] or proto['threshold'] != .5):
                raise ValueError('Require complete continuous-prior native audit')
            if local_confidence and ar.get('matched_reference_steps_checked') != 300:
                raise ValueError('Require exact matched local-confidence observations')
            for path in (a.run/'protocol.json', a.run/'report.json'):
                if ar['source_sha256'].get(str(path)) != bindings[str(path)]:
                    raise ValueError('Continuous-prior source run not bound')
            evidence = Path(proto['source_evidence']); ep = read(evidence/'protocol.json')
            
            
            for cohort, splits in (('validation_paired', {'validation'}), ('development_hn', {'hn_development', 'train'})):
                path = evidence/ep['cohort_manifests'][cohort]
                rows = [json.loads(s) for s in raw(path).splitlines() if s.strip()]
                if (ar['source_sha256'].get(str(path)) != bindings[str(path)]
                        or ep['output_sha256'][path.name] != bindings[str(path)] or len(rows) != 100):
                    raise ValueError('Continuous-prior original cohort not bound')
                for r in rows:
                    if r['sample_id'] in sources or r['split'] not in splits:
                        raise ValueError('Overlapping or forbidden subgroup source')
                    sources[r['sample_id']] = r
        cp=a.audit/'native_cases.json'
        region_cache=None
        if not cp.exists():
            cp=a.audit/'native_pixel_cases.json'
            region_cache=read(a.run/'cache_inventory.json')
            for item in region_cache.values():
                if ar['source_sha256'].get(item['path']) not in (None,item['sha256']):
                    raise ValueError('Historical feature-cache identity changed')
        cases=read(cp); baseline=cases['C_frozen']; ids={r['sample_id'] for r in baseline}
        if len(baseline)!=200 or len(ids)!=200 or sum(r['role']=='bug' for r in baseline)!=50:
            raise ValueError('Require original Dev200 with50 Bug')
        expected_arms = ({'C_frozen','H_hard_local_input_reference','S_soft_local_input'} if local_confidence
                         else {'C_frozen','B_binary_prior_reference','R_continuous_prior'})
        if continuous and (set(sources) != ids or set(cases) != expected_arms):
            raise ValueError('Incomplete continuous-prior method comparison')
        parent=Path(proto.get('teacher_run') or proto['source_run'])
        predictions={}
        for arm in cases:
            if continuous:
                if len(cases[arm]) != 200 or {r['sample_id'] for r in cases[arm]} != ids:
                    raise ValueError('Incomplete method comparison')
                predictions[arm] = {}
                continue
            if arm=='C_frozen':
                paths=[parent/'C_zero_motion/checkpoint-0003/evaluation'/c/'predictions.jsonl' for c in ('validation_paired','development_hn')]
            else: paths=[a.run/arm/'predictions.jsonl']
            rows=[json.loads(line) for path in paths for line in raw(path).splitlines() if line.strip()]
            for path in paths:
                if ar['source_sha256'].get(str(path)) not in (None,bindings[str(path)]):
                    raise ValueError('Prediction differs from independent audit')
            if len({r['sample_id'] for r in rows})!=len(rows): raise ValueError('Duplicate source predictions')
            predictions[arm]={r['sample_id']:r for r in rows if r['sample_id'] in ids}
            if set(predictions[arm])!=ids or len(cases[arm])!=200 or {r['sample_id'] for r in cases[arm]}!=ids:
                raise ValueError('Incomplete method comparison')
        results={}; reference_truth={}
        for arm, metadata in cases.items():
            status(status='scoring_frozen_subgroups', arm=arm)
            pixels=defaultdict(list); temporal=defaultdict(list)
            for index, case in enumerate(metadata):
                sid=case['sample_id']
                token=metric_token(sid,region_cache)
                frames=read(a.audit/'pixel_metrics'/f'{token}_{arm}.json')
                pred=prediction_from_native_counts(sources[sid],frames) if continuous else predictions[arm][sid]
                validate_frames(frames,pred)
                predictions[arm][sid]=pred
                if sid in reference_truth and pred['frame_targets'] != reference_truth[sid]: raise ValueError('Changed temporal labels')
                reference_truth[sid]=pred['frame_targets']
                tags=['all','root:'+case['root'],'camera:'+case['camera_condition'],'fidelity:'+case['fidelity'],'role:'+case['role']]
                if continuous:
                    source=sources[sid]
                    if (case['role'],case['root'],case['camera_condition']) != (source['role'],source['root_cause_family'],source['camera_condition']):
                        raise ValueError('Source subgroup identity differs')
                    if source['role'] == 'hard_negative':
                        category=source.get('hard_negative_category')
                        if not category: raise ValueError('Missing source Hard Negative category')
                        tags.append('hn_category:'+category)
                for tag in tags: temporal[tag].append(pred)
                for frame in frames:
                    row=dict(frame,sample_id=sid)
                    for tag in [*tags,'area:'+frame['target_area_bin']]: pixels[tag].append(row)
                if index%20==0: status(video_index=index+1)
            results[arm]=dict(pixel={k:pixel_stratum(v) for k,v in pixels.items()},temporal={k:temporal_strata(v) for k,v in temporal.items()})
            if continuous:
                expected=producer['results'][arm]['development']['event_metrics']['3']
                actual=results[arm]['temporal']['all']['endpoint_metrics']['3']
                if any(actual[k] != expected[k] for k in ('tp','fp','fn','f1')):
                    raise ValueError('Native decisions differ from original endpoint results')
        for path in list(bindings): raw(path)
        save('report.json',dict(completed=True, threshold=.5, tolerance_frames=3, data='Repeated synthetic Dev200 only',
            new_predictions=False, training_updates=0, test_used=False, real_gt_used=False, automatic_promotion=False,
            input_decisions='Audited native mask counts; continuous probabilities NOT reconstructed' if continuous else 'Original prediction scores and native counts',
            original_source_split_fields={role:{split:sum(r['role']==role and r['split']==split for r in sources.values())
                for split in sorted({r['split'] for r in sources.values() if r['role']==role})}
                for role in sorted({r['role'] for r in sources.values()})} if continuous else None,
            conventions=['Strict frame/pixel counts remain strict; +/-3 applies only to BOTH event endpoints.',
                'Area bins use original target pixels, for evaluation only; no inference ROI.',
                'Pixel IoU and Boundary F1 averaged per case, then cases equally; micro counts separate.',
                'Pixel intersection recall is separate from any-prediction frame recall.',
                'Strict spatial90 means BOTH native pixel precision/recall>=90%; not a box score, boundary-tolerant metric or new gate.',
                'Normal gaps are only spans between two target events. No gap fill or threshold adjustment.',
                'No new acceptance gates; subgroup diagnosis cannot replace original frozen gates.',
                'Development reuse, only5 paired scenes and upstream HN exposure prohibit real-world claims.'],
            source_sha256=bindings, code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (out/'code').glob('*.py')}, results=results))
        status(status='completed')
    except BaseException as error:
        status(status='failed', error=repr(error)); raise


if __name__=='__main__': main()
