
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from report_native_tolerant_events import intervals, match_events, evaluate
from report_native_detail_subgroups import validate_frames

STATES=('no_shared_source_frame','shared_frame_wrong_region','partial_tube_overlap','strict_tube_precision90_recall90')
REGION_PROTOCOLS=('native_local_region_authenticity_v1','native_local_correspondence_control_v1')


def row_from_audited_native_counts(source,frames):
    
    if len(frames)!=source['frame_count']:
        raise ValueError('Incomplete original source coverage')
    row=dict(sample_id=source['sample_id'],frame_count=source['frame_count'],fps=source['fps'],
        frame_targets=[int(f['target_positive_pixels']>0) for f in frames],
        frame_probabilities=[float(f['predicted_positive_pixels']>0) for f in frames])
    validate_frames(frames,row)
    return row


def joint_events(row, frames, tolerance=3):
    validate_frames(frames,row)
    gt=intervals(row['frame_targets']);pred=intervals([p>=.5 for p in row['frame_probabilities']])
    matches=dict(match_events(gt,pred,tolerance));events=[]
    for i,(start,end) in enumerate(gt):
        target=sum(frames[t]['target_positive_pixels'] for t in range(start,end+1))
        if i not in matches:
            events.append(dict(gt=[start,end],prediction=None,matched=False,target_pixels_over_time=target,
                intersection_pixels_over_time=0,predicted_pixels_over_time=0,strict_native_tube_iou=0.,state='unmatched_gt'))
            continue
        a,b=pred[matches[i]];left,right=max(start,a),min(end,b)
        intersection=sum(frames[t]['pixels']['tp'] for t in range(left,right+1))
        predicted=sum(frames[t]['predicted_positive_pixels'] for t in range(a,b+1))
        if min(predicted,target)<=0 or intersection>min(predicted,target):raise ValueError('Invalid matched tube counts')
        if left>right:state='no_shared_source_frame'
        elif intersection==0:state='shared_frame_wrong_region'
        elif 10*intersection>=9*predicted and 10*intersection>=9*target:state='strict_tube_precision90_recall90'
        else:state='partial_tube_overlap'
        events.append(dict(gt=[start,end],prediction=[a,b],matched=True,state=state,
            onset_error_frames=a-start,offset_error_frames=b-end,shared_source_frames=max(0,right-left+1),
            target_pixels_over_time=target,predicted_pixels_over_time=predicted,intersection_pixels_over_time=intersection,
            strict_native_tube_iou=intersection/(target+predicted-intersection),
            strict_native_tube_precision=intersection/predicted,strict_native_tube_recall=intersection/target))
    return dict(sample_id=row['sample_id'],gt_events=len(gt),predicted_events=len(pred),matched_events=len(matches),
        false_events=len(pred)-len(matches),missed_events=len(gt)-len(matches),events=events)


def aggregate(cases):
    matched=[e for c in cases for e in c['events'] if e['matched']]
    all_gt=[e for c in cases for e in c['events']]
    states={s:sum(e['state']==s for e in matched) for s in STATES}
    tp=sum(c['matched_events'] for c in cases);fp=sum(c['false_events'] for c in cases);fn=sum(c['missed_events'] for c in cases)
    positive=[c for c in cases if c['events']]
    return dict(videos=len(cases),gt_events=len(all_gt),predicted_events=sum(c['predicted_events'] for c in cases),
        temporal_tp=tp,temporal_fp=fp,temporal_fn=fn,
        unchanged_endpoint_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        temporal_matched_states=states,unmatched_gt_events=fn,
        mean_all_gt_event_strict_native_tube_iou=sum(e['strict_native_tube_iou'] for e in all_gt)/len(all_gt) if all_gt else None,
        mean_case_all_gt_event_strict_native_tube_iou=sum(sum(e['strict_native_tube_iou'] for e in c['events'])/len(c['events']) for c in positive)/len(positive) if positive else None,
        conditional_temporal_matched_mean_strict_native_tube_iou=sum(e['strict_native_tube_iou'] for e in matched)/len(matched) if matched else None,
        normal_videos=sum(c['gt_events']==0 for c in cases),
        normal_false_positive_videos=sum(c['gt_events']==0 and c['predicted_events']>0 for c in cases))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','audit','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--evidence',type=Path,help='Required only for legacy independently reconstructed regional audits')
    p.add_argument('--launch',action='store_true');p.add_argument('--worker',action='store_true');a=p.parse_args();out=a.output
    if not a.worker:
        out.mkdir(parents=True,exist_ok=False);code=out/'code';code.mkdir()
        for name in (Path(__file__).name,'report_native_tolerant_events.py','report_native_detail_subgroups.py'):
            shutil.copy2(Path(__file__).with_name(name),code/name)
        cmd=[sys.executable,str(code/Path(__file__).name),*[x for x in sys.argv[1:] if x!='--launch'],'--worker']
        if a.launch:
            with (out/'worker.log').open('x') as log:proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            (out/'launch.json').write_text(json.dumps(dict(pid=proc.pid,command=cmd),indent=2));print(json.dumps(dict(pid=proc.pid,output=str(out))));return
        os.execv(sys.executable,cmd)
    state=dict(pid=os.getpid());bindings={}
    def save(name,value):
        path=out/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    def status(**kw):state.update(kw);state['updated_unix_s']=time.time();save('status.json',state)
    def raw(path):
        path=Path(path);data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
        if str(path) in bindings and bindings[str(path)]!=digest:raise ValueError('Source mutated')
        bindings[str(path)]=digest;return data
    def read(path):return json.loads(raw(path))
    try:
        while True:
            s=json.loads((a.audit/'status.json').read_text())
            if s['status']=='completed':break
            if s['status']=='failed':raise RuntimeError('Independent native audit failed')
            proc=Path('/proc')/str(s['pid'])/'cmdline';cmd=proc.read_bytes().decode().replace('\0',' ') if proc.exists() else ''
            if str(a.audit) not in cmd:
                if json.loads((a.audit/'status.json').read_text())['status']=='completed':break
                raise RuntimeError('Audit producer missing; no stale waiting')
            status(status='waiting_for_verified_native_audit',producer_pid=s['pid']);time.sleep(15)
        ar=read(a.audit/'report.json');proto=read(a.run/'protocol.json');region=proto['protocol'] in REGION_PROTOCOLS
        local_confidence=proto['protocol']=='native_local_confidence_control_v1'
        continuous=local_confidence or proto['protocol']=='native_continuous_prior_control_v1'
        counts_mode=region or continuous
        region_sources={};region_cache=None;producer=None
        if region:
            producer=read(a.run/'report.json')
            if (a.evidence is None or not ar.get('passed_native_reconstruction') or not ar['completed']
                    or ar['development_videos']!=200 or ar['head_replay_videos']!=400
                    or not producer['completed'] or producer['execution_smoke']
                    or any(v['test_used'] or v['real_gt_used'] for v in (ar,producer,proto))
                    or proto['threshold']!=.5):raise ValueError('Require full independent regional reconstruction')
            for path in (a.run/'protocol.json',a.run/'report.json'):
                if ar['source_sha256'].get(str(path))!=bindings[str(path)]:raise ValueError('Regional source run not bound')
            ep=read(a.evidence/'protocol.json')
            for cohort in ('validation_paired','development_hn'):
                path=a.evidence/ep['cohort_manifests'][cohort]
                rows=[json.loads(s) for s in raw(path).splitlines() if s.strip()]
                if ar['source_sha256'].get(str(path))!=bindings[str(path)] or ep['output_sha256'][path.name]!=bindings[str(path)]:
                    raise ValueError('Regional original cohort hash differs')
                for r in rows:
                    if r['sample_id'] in region_sources or r['split']=='test':raise ValueError('Overlapping or forbidden cohort')
                    region_sources[r['sample_id']]=r
            cache_path=a.run/'cache_inventory.json';region_cache=read(cache_path)
            if ar['source_sha256'].get(str(cache_path))!=bindings[str(cache_path)]:raise ValueError('Regional frame key inventory differs')
            for sid in region_sources:
                item=region_cache[sid]
                if ar['source_sha256'].get(item['path'])!=item['sha256']:raise ValueError('Regional metric token not bound')
            cases=read(a.audit/'native_pixel_cases.json')
        else:
            if (not ar['passed'] or not ar['completed'] or ar['execution_smoke'] or ar['test_used'] or ar['real_gt_used']
                    or ar['source_run']!=str(a.run) or proto['threshold']!=.5):raise ValueError('Require complete frozen full-Dev audit')
            cases=read(a.audit/'native_cases.json')
            if continuous:
                producer=read(a.run/'report.json')
                flag='local_confidence_control' if local_confidence else 'continuous_prior_control'
                if (not ar.get(flag) or ar['training_steps_checked']!=300
                        or ar['independent_native_video_replays']!=200 or ar['fresh_frozen_C_replays']!=200
                        or ar['independent_directional_intersections']!=200 or not producer['completed']
                        or producer['execution_smoke'] or producer['test_used'] or producer['real_gt_used']):
                    raise ValueError('Require complete continuous-prior native audit')
                if local_confidence and ar.get('matched_reference_steps_checked')!=300:
                    raise ValueError('Require exact matched local-confidence observations')
                for path in (a.run/'protocol.json',a.run/'report.json'):
                    if ar['source_sha256'].get(str(path))!=bindings[str(path)]:raise ValueError('Prior source run not bound')
                evidence=Path(proto['source_evidence']);ep=read(evidence/'protocol.json')
                for cohort in ('validation_paired','development_hn'):
                    path=evidence/ep['cohort_manifests'][cohort]
                    records=[json.loads(s) for s in raw(path).splitlines() if s.strip()]
                    if (ar['source_sha256'].get(str(path))!=bindings[str(path)]
                            or ep['output_sha256'][path.name]!=bindings[str(path)]):
                        raise ValueError('Continuous-prior original cohort not bound')
                    for r in records:
                        if r['sample_id'] in region_sources or r['split']=='test':raise ValueError('Overlapping/forbidden source')
                        region_sources[r['sample_id']]=r
        baseline=cases['C_frozen'];ids={r['sample_id'] for r in baseline}
        if local_confidence and set(cases)!={'C_frozen','H_hard_local_input_reference','S_soft_local_input'}:
            raise ValueError('Unexpected local-confidence methods')
        if len(baseline)!=200 or len(ids)!=200 or sum(r['role']=='bug' for r in baseline)!=50:raise ValueError('Wrong Dev population')
        if counts_mode and set(region_sources)!=ids:raise ValueError('Audited cohort identities differ')
        parent=Path(proto['source_run'] if region else proto['teacher_run']);results={};all_cases={};reference_truth={}
        for arm,case_meta in cases.items():
            status(status='scoring_frozen_joint_events',arm=arm)
            paths=([parent/'C_zero_motion/checkpoint-0003/evaluation'/c/'predictions.jsonl' for c in ('validation_paired','development_hn')]
                   if arm=='C_frozen' else [a.run/arm/'predictions.jsonl'])
            predictions={}
            for path in ([] if counts_mode else paths):
                records=[json.loads(s) for s in raw(path).splitlines() if s.strip()]
                if ar['source_sha256'].get(str(path))!=bindings[str(path)]:raise ValueError('Prediction not bound by native audit')
                for r in records:
                    if r['sample_id'] in ids:
                        if r['sample_id'] in predictions:raise ValueError('Duplicate prediction')
                        predictions[r['sample_id']]=r
            if (not counts_mode and set(predictions)!=ids) or {r['sample_id'] for r in case_meta}!=ids:raise ValueError('Incomplete comparison')
            groups=defaultdict(list);records=[]
            for meta in case_meta:
                sid=meta['sample_id']
                token=region_cache[sid]['sha256'][:16] if region else hashlib.sha256(sid.encode()).hexdigest()[:16]
                frames=read(a.audit/'pixel_metrics'/f'{token}_{arm}.json')
                row=row_from_audited_native_counts(region_sources[sid],frames) if counts_mode else predictions[sid]
                if counts_mode:predictions[sid]=row
                if sid in reference_truth and row['frame_targets']!=reference_truth[sid]:raise ValueError('GT changed')
                reference_truth[sid]=row['frame_targets']
                result=joint_events(row,frames,3);records.append(dict(result,root=meta['root'],camera_condition=meta['camera_condition']))
                for key in ('all','root:'+meta['root'],'camera:'+meta['camera_condition'],'role:'+meta['role']):groups[key].append(result)
            results[arm]={key:aggregate(rows) for key,rows in groups.items()};all_cases[arm]=records
            original=evaluate(list(predictions.values()),3);current=results[arm]['all']
            if any(current['temporal_'+key]!=original[key] for key in ('tp','fp','fn')) or current['unchanged_endpoint_f1']!=original['f1']:
                raise ValueError('Original endpoint metric changed')
            if counts_mode:
                expected=producer['results'][arm]['development']['event_metrics']['3']
                if any(original[k]!=expected[k] for k in ('tp','fp','fn','f1')):
                    raise ValueError('Reconstructed native decisions differ from original endpoint results')
        for path in list(bindings):raw(path)
        save('cases.json',all_cases)
        save('report.json',dict(completed=True,results=results,source_sha256=bindings,
            code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'code').glob('*.py')},
            scope='Supplemental joint native counts on reused Dev200; no new model/inference/training/selection',
            input_decisions='Audited native mask counts; continuous probabilities NOT reconstructed' if counts_mode else 'Audit-bound original prediction scores plus native counts',
            threshold=.5,tolerance_frames=3,test_used=False,real_gt_used=False,training_updates=0,model_inference=False,automatic_promotion=False,
            conventions=['Original maximum-cardinality endpoint matches reused unchanged; no spatial-aware rematching.',
                'Tube intersection uses BOTH matched intervals AND the exact same source-frame pixel intersections.',
                'No shared source frame is separate from wrong spatial region: +/-3 allows timing shifts without overlap.',
                'Strict tube counts use no temporal shift, dilation, bbox filling or object assignment; not a new acceptance gate.',
                'Missed GT events retain IoU0; normal videos do not get invented perfect localization.',
                'Case means, event means and matched-only means reported separately; original synthetic GT fidelity limitations remain.']))
        status(status='completed')
    except BaseException as error:status(status='failed',error=repr(error));raise


if __name__=='__main__':main()
