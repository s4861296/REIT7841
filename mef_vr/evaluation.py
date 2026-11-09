
import csv,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from . import runtime as rt

def count_metrics(c):
    if any(type(c.get(k,0)) is not int or c.get(k,0)<0 for k in ('tp','tn','fp','fn')):
        raise ValueError('Confusion counts must be nonnegative integers')
    tp,tn,fp,fn=[int(c.get(k,0)) for k in ('tp','tn','fp','fn')]
    div=lambda a,b:a/b if b else None
    return dict(tp=tp,tn=tn,fp=fp,fn=fn,n=tp+tn+fp+fn,
        accuracy=div(tp+tn,tp+tn+fp+fn),precision=div(tp,tp+fp),recall=div(tp,tp+fn),
        specificity=div(tn,tn+fp),f1=div(2*tp,2*tp+fp+fn))

def score_records(records):
    ids=[r['sample_id'] for r in records]
    if not records or len(ids)!=len(set(ids)):raise ValueError('Empty cohort or duplicate sample IDs')
    c=Counter();failed=[]
    for r in records:
        y,p=r['label'],r['predicted_label']
        if type(y) is not int or y not in (0,1):raise ValueError('Invalid label')
        if p is None:failed.append(r['sample_id']);continue
        if type(p) is not int or p not in (0,1):raise ValueError('Invalid prediction')
        c[('tp' if p else 'fn') if y else ('fp' if p else 'tn')]+=1
    m=count_metrics(c)
    return dict(m,target_count=len(records),valid_count=m['n'],failed_count=len(failed),failed_ids=failed,
        accuracy_all_cases=(c['tp']+c['tn'])/len(records),
        metric_scope='completed predictions; execution failures explicitly listed, not labeled Clean')

def metrics_file(path,out):
    data=rt.read(path);out=Path(out)
    if out.exists():raise FileExistsError('Use a fresh metrics directory')
    out.mkdir(parents=True);rows=[]
    for exp in data['experiments']:
        if 'records' in exp: m=score_records(exp['records'])
        elif 'counts' in exp:m=count_metrics(exp['counts'])
        else:continue
        expected=exp.get('expected_counts')
        if expected and any(m[k]!=expected[k] for k in ('tp','tn','fp','fn')):
            raise ValueError('Archived aggregate disagrees with sample-level predictions: '+exp['id'])
        rows.append(dict(experiment=exp['id'],dataset=exp['dataset'],method=exp['method'],
            protocol=exp.get('protocol',''),evidence_level='per_video' if 'records' in exp else 'aggregate_counts',**m))
    columns=['experiment','dataset','method','protocol','evidence_level','n','tp','tn','fp','fn',
        'accuracy','precision','recall','f1','specificity','target_count','valid_count','failed_count','accuracy_all_cases']
    with (out/'classification.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
    rt.save(out/'classification.json',rows)
    localization=[]
    for exp in data.get('localization',[]):
        r=dict(exp)
        for field in ('frame_counts','event_counts'):
            if field in exp:
                c=exp[field];den=2*c['tp']+c['fp']+c['fn']
                r[field.replace('_counts','_f1_recomputed')]=2*c['tp']/den if den else None
        localization.append(r)
    rt.save(out/'localization.json',localization)
    
    development=[]
    if data.get('development_frame_records'):
        from report_native_tolerant_events import evaluate
        for exp in data['development_frame_records']:
            e=evaluate(exp['records'],3)
            development.append(dict(id=exp['id'],event_tolerance_original_frames=3,event_metrics=e))
    rt.save(out/'development_events.json',development)
    rt.save(out/'receipt.json',dict(source=str(path),source_sha256=rt.sha(path),classification_rows=len(rows),
        no_model_inference=True,no_threshold_fit=True,localization_rows=len(localization)))
    print(json.dumps(dict(classification_rows=len(rows),output=str(out)),ensure_ascii=False))
    return rows

def evaluate(args):
    sources=rt.read(rt.ROOT/f'data/manifests/{args.split}.json')
    truth={r['sample_id']:r['label'] for r in rt.read(rt.ROOT/f'data/labels/{args.split}.json')}
    if args.ids:
        unknown=set(args.ids)-{r['sample_id'] for r in sources}
        if unknown:raise ValueError('Unknown sample IDs: '+str(unknown))
        sources=[r for r in sources if r['sample_id'] in args.ids]
    out=args.output.resolve()
    if out.exists() and not args.resume:raise FileExistsError('Output exists; use --resume only for the same job')
    out.mkdir(parents=True,exist_ok=True)
    contract=dict(split=args.split,ids=[r['sample_id'] for r in sources],ablations=args.ablations,
        keep_features=args.keep_features,config_sha256=rt.sha(rt.ROOT/'configs/detector.json'))
    if (out/'job.json').exists():
        if rt.read(out/'job.json')!=contract:raise ValueError('Resume configuration differs')
    else:rt.save(out/'job.json',contract)
    records=[];arms={}
    for source in sources:
        sid=source['sample_id'];key=hashlib.sha256(sid.encode()).hexdigest()[:20];folder=out/'cases'/key
        try:
            if (folder/'completed.json').exists():
                done=rt.read(folder/'completed.json')
                if done['source_video_sha256']!=source['video_sha256'] or done['prediction_sha256']!=rt.sha(folder/'prediction.json'):
                    raise ValueError('Resume input/prediction hash differs')
                answer=rt.read(folder/'prediction.json')
            elif folder.exists():raise RuntimeError('Prior incomplete attempt retained; use a fresh batch directory to retry')
            else:answer=rt.detect(source,folder,ablations=args.ablations,keep_features=args.keep_features)
            records.append(dict(sample_id=sid,label=truth[sid],predicted_label=int(answer['predicted_label']),status='completed'))
            if args.ablations:
                for arm,p in rt.read(folder/'interventions.json').items():
                    arms.setdefault(arm,[]).append(dict(sample_id=sid,label=truth[sid],predicted_label=p))
        except Exception as error:
            records.append(dict(sample_id=sid,label=truth[sid],predicted_label=None,status='failed',error=repr(error)))
        rt.save(out/'progress.json',dict(completed=len(records),total=len(sources),failed=sum(r['predicted_label'] is None for r in records)))
    experiments=[dict(id='fresh_complete',dataset=args.split,method='MEF-VR complete',protocol='fresh original-video inference',records=records)]
    for arm,rows in arms.items():
        present={r['sample_id'] for r in rows}
        rows += [dict(sample_id=r['sample_id'],label=r['label'],predicted_label=None) for r in records if r['sample_id'] not in present]
        experiments.append(dict(id='fresh_'+arm,dataset=args.split,method=arm,protocol='fixed-weight intervention',records=rows))
    rt.save(out/'all_results.json',dict(experiments=experiments))
    metric_out=out/('metrics_'+str(int(__import__('time').time())))
    metrics_file(out/'all_results.json',metric_out)
    if any(r['predicted_label'] is None for r in records):
        raise RuntimeError('Batch contains failed inputs; inspect all_results.json. No failures were treated as Clean.')
