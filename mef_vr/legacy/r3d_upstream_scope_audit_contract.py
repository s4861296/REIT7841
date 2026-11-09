
from collections import Counter
import hashlib
import math
import numpy as np


def require(condition, message):
    if not condition: raise ValueError(message)


def runs(flags):
    a=np.asarray(flags,dtype=bool)
    switches=np.diff(np.r_[False,a,False].astype(np.int8))
    return list(zip(np.flatnonzero(switches==1).tolist(),(np.flatnonzero(switches==-1)-1).tolist()))


def matching(target, predicted, tolerance=3):
    edges=[sorted([j for j,b in enumerate(predicted) if max(abs(a[0]-b[0]),abs(a[1]-b[1]))<=tolerance],
                  key=lambda j:(abs(a[0]-predicted[j][0])+abs(a[1]-predicted[j][1]),j)) for a in target]
    left=[-1]*len(target);right=[-1]*len(predicted)
    for root in range(len(target)):
        queue=[root];reached={root};parents={};found=-1
        for current in queue:
            for j in edges[current]:
                if j in parents:continue
                parents[j]=current
                if right[j]<0:found=j;break
                if right[j] not in reached:reached.add(right[j]);queue.append(right[j])
            if found>=0:break
        while found>=0:
            owner=parents[found];previous=left[owner];left[owner]=found;right[found]=owner;found=previous
    pairs=[(i,j) for i,j in enumerate(left) if j>=0]
    
    owners={}
    def augment(i,visited):
        for j in edges[i]:
            if j in visited:continue
            visited.add(j)
            if j not in owners or augment(owners[j],visited):owners[j]=i;return True
        return False
    for i in range(len(target)):augment(i,set())
    require(len(pairs)==len(owners),'Matching does not have maximum cardinality')
    return pairs


def score(rows):
    c={k:0 for k in ('frame_tp','frame_fp','frame_fn','frame_tn','video_tp','video_fp','video_fn','video_tn',
        'event_tp','event_fp','event_fn','short_events','short_hits','normal_gap_frames','normal_gap_false_frames',
        'clean_videos','clean_false_videos','hn_videos','hn_false_videos')}
    seen=set();details=[]
    for row in rows:
        sid=row['sample_id'];y=np.asarray(row['frame_targets']);s=np.asarray(row['frame_probabilities'],dtype=np.float64)
        require(sid not in seen and y.ndim==s.ndim==1 and len(y)==len(s)==row['frame_count']>0,'Duplicate or incomplete source')
        seen.add(sid);require(np.isin(y,[0,1]).all() and np.isfinite(s).all() and ((s>=0)&(s<=1)).all(),'Invalid scores or labels')
        y=y.astype(bool);p=s>=.295
        c['frame_tp']+=int((y&p).sum());c['frame_fp']+=int((~y&p).sum());c['frame_fn']+=int((y&~p).sum());c['frame_tn']+=int((~y&~p).sum())
        c['video_tp']+=int(y.any() and p.any());c['video_fp']+=int(not y.any() and p.any())
        c['video_fn']+=int(y.any() and not p.any());c['video_tn']+=int(not y.any() and not p.any())
        target,predicted=runs(y),runs(p);pairs=matching(target,predicted)
        c['event_tp']+=len(pairs);c['event_fp']+=len(predicted)-len(pairs);c['event_fn']+=len(target)-len(pairs)
        c['short_events']+=sum(b-a<3 for a,b in target);c['short_hits']+=sum(target[i][1]-target[i][0]<3 for i,_ in pairs)
        gap=np.zeros(len(y),bool)
        if target:gap[target[0][0]:target[-1][1]+1]=~y[target[0][0]:target[-1][1]+1]
        c['normal_gap_frames']+=int(gap.sum());c['normal_gap_false_frames']+=int((gap&p).sum())
        if row['role']=='clean_or_fixed':c['clean_videos']+=1;c['clean_false_videos']+=int(p.any())
        if row['role']=='hard_negative':c['hn_videos']+=1;c['hn_false_videos']+=int(p.any())
        details.append(dict(sample_id=sid,matches=[dict(target=list(target[i]),prediction=list(predicted[j])) for i,j in pairs],
            target_events=len(target),predicted_events=len(predicted),false_normal_gap_frames=int((gap&p).sum())))
    def div(a,b):return a/b if b else None
    m=dict(c,videos=len(rows),threshold=.295,tolerance_frames=3,
        video_accuracy=div(c['video_tp']+c['video_tn'],len(rows)),
        frame_f1=div(2*c['frame_tp'],2*c['frame_tp']+c['frame_fp']+c['frame_fn']),
        frame_recall=div(c['frame_tp'],c['frame_tp']+c['frame_fn']),
        endpoint_event_f1=div(2*c['event_tp'],2*c['event_tp']+c['event_fp']+c['event_fn']),
        short_event_recall=div(c['short_hits'],c['short_events']))
    return m,details


def coverage(n,window=32,stride=8):
    require(type(n) is int and n>0,'Positive source length required')
    starts=[0] if n<=window else list(range(0,n-window+1,stride))
    if n>window and starts[-1]!=n-window:starts.append(n-window)
    counts=[0]*n
    for start in starts:
        for t in range(start,min(start+window,n)):counts[t]+=1
    return starts,counts


def frame_labels(record):
    n=record['frame_count'];answer=[0]*n
    if record['role']=='bug':
        for event in record['answer']['events']:
            a,b=int(event['start_frame']),int(event['end_frame'])
            require(a==event['start_frame'] and b==event['end_frame'] and a<=b,'Invalid event frame indices')
            for t in range(max(0,a),min(n-1,b)+1):answer[t]=1
    return answer


def parameter_groups(state, exclude_buffers=False):
    counts=Counter()
    for name,value in state.items():
        if exclude_buffers and name.endswith(('running_mean','running_var','num_batches_tracked')):continue
        counts[name.split('.')[0]]+=value.numel()
    return dict(counts)


def check_gradient_groups(groups, expected, active):
    require(set(groups)==set(expected),'Gradient group inventory differs')
    for key,number in expected.items():
        row=groups[key];selected=number if key in active else 0
        require(row['parameters']==number and row['trainable']==selected and row['with_gradient']==selected,'Missing or undeclared parameter gradients')
        require(row['all_finite'] is True and 0<=row['nonzero_gradient']<=selected,'Nonfinite or impossible gradients')
        require(math.isfinite(row['squared_gradient_norm']) and row['squared_gradient_norm']>=0,'Invalid gradient norm')
        if not selected:require(row['squared_gradient_norm']==0,'Frozen group gradient norm nonzero')


def check_history(rows, protocol, pairs, negatives, orders, bone_groups, head_groups):
    require(len(rows)==600,'Complete600-update history required')
    arm_names=list(protocol['arms']);total_frames=0;uses=Counter()
    for index,row in enumerate(rows):
        arm=arm_names[index//300];step=index%300+1;epoch=(step-1)//100+1;position=(step-1)%100
        pair_id=orders[epoch-1][position];pair=pairs[pair_id];negative=negatives[protocol['hn_clip_pairing'][pair_id]]
        source_ids=[r['sample_id'] for r in pair]+[negative['sample_id']]
        require((row['arm'],row['step'],row['epoch'],row['pair_id'],row['hn_id'])==(arm,step,epoch,pair_id,negative['sample_id']),'Training order/update mismatch')
        require(row['source_ids']==source_ids and row['optimizer_updates']==1,'Wrong source triplet or update count')
        bug=next(r for r in pair if r['role']=='bug');clean=next(r for r in pair if r['role']=='clean_or_fixed')
        require(all(bug[k]==clean[k] for k in ('frame_count','fps','height','width')),'Pair frame/FPS/shape mismatch')
        starts,cov=coverage(bug['frame_count']);hn_starts,hn_cov=coverage(negative['frame_count'])
        require(row['pair_coverage']==cov and row['hn_coverage']==hn_cov,'Skipped/repeated/padded source frame contribution')
        require(row['paired_windows']==len(starts) and row['hn_windows']==len(hn_starts),'Wrong original window count')
        unique=2*bug['frame_count']+negative['frame_count']
        require(row['unique_source_frames']==unique and row['window_frame_uses']==2*sum(cov)+sum(hn_cov),'Original frame count mismatch')
        positive=max(1,sum(frame_labels(bug)))
        expected_weight=max(protocol['global_positive_weight'],min(100.,(2*bug['frame_count']-positive)/positive))
        require(row['pair_positive_weight']==expected_weight,'Positive loss weight changed')
        for field in ('classification_loss','ranking_loss','hard_negative_loss','gradient_norm_before_joint_clip','seconds'):
            require(math.isfinite(row[field]) and row[field]>=0,'Nonfinite/negative training quantity')
        grads=row['gradients_before_clip'];active={'stem','layer1'} if arm=='B_early_r3d' else set()
        check_gradient_groups(grads['backbone'],bone_groups,active);check_gradient_groups(grads['head'],head_groups,set(head_groups))
        squared=sum(v['squared_gradient_norm'] for group in grads.values() for v in group.values())
        require(math.isclose(row['gradient_norm_before_joint_clip']**2,squared,rel_tol=1e-5,abs_tol=1e-8),'Joint optimizer gradient norm inconsistent')
        total_frames+=unique;uses.update((arm,sid) for sid in source_ids)
    require(len(uses)==600 and set(uses.values())=={3},'Every Train source must appear3times in each arm')
    return dict(updates=600,full_video_uses=sum(uses.values()),original_frame_uses=total_frames,
        unique_training_videos_per_arm=300,each_source_uses_per_arm=3,all_coverage_and_gradient_scopes_passed=True)


def state_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        array=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str((array.dtype,tuple(array.shape))).encode());h.update(array.numpy().tobytes())
    return h.hexdigest()


def check_checkpoint(state, initial_bone, initial_head, arm, epoch, protocol_sha):
    import torch
    require((state['arm'],state['epoch'],state['step'],state['protocol_sha256'])==(arm,epoch,100*epoch,protocol_sha),'Checkpoint identity/budget changed')
    selected=[r['name'] for r in state['selected_backbone_parameters']]
    expected=[k for k in initial_bone if k.split('.')[0] in ('stem','layer1') and not k.endswith(('running_mean','running_var','num_batches_tracked'))] if arm=='B_early_r3d' else []
    require(selected==expected,'Checkpoint selected scope differs')
    require(state['selected_backbone_parameters']==[dict(name=k,shape=list(initial_bone[k].shape),numel=initial_bone[k].numel()) for k in expected],'Selected parameter dimensions differ')
    require(state['scope']==('stem_layer1_and_head' if arm=='B_early_r3d' else 'head_only'),'Scope identifier differs')
    changed=[]
    for branch,initial in (('backbone',initial_bone),('head',initial_head)):
        require(list(state[branch])==list(initial),'Missing/reordered model parameters/buffers')
        for name,value in state[branch].items():
            require(value.shape==initial[name].shape and value.dtype==initial[name].dtype and torch.isfinite(value).all(),'Invalid checkpoint tensor')
            if branch=='backbone' and name not in selected:require(torch.equal(value,initial[name]),'Frozen parameter or BN running buffer changed')
            if branch=='backbone' and not torch.equal(value,initial[name]):changed.append(name)
    require([state_digest(state['backbone']),state_digest(state['head'])]==state['checkpoint_model_hashes'],'Checkpoint state digest inconsistent')
    if arm=='B_early_r3d':require({k.split('.')[0] for k in changed}=={'stem','layer1'},'Selected early layers never updated')
    require(any(not torch.equal(state['head'][k],initial_head[k]) for k in initial_head),'Temporal head never updated')
    values=[state['backbone'][k] for k in selected]+list(state['head'].values())
    groups=state['optimizer']['param_groups'];require(len(groups)==1,'One jointly clipped optimizer group required')
    group=groups[0];ids=group['params']
    require(len(ids)==len(set(ids))==len(values) and set(ids)==set(state['optimizer']['state']),'Optimizer state/ownership differs')
    require(group['lr']==3e-5 and group['weight_decay']==1e-4 and tuple(group['betas'])==(.9,.999),'Optimizer settings changed')
    for ident,value in zip(ids,values):
        item=state['optimizer']['state'][ident]
        require(float(item['step'])==100*epoch,'Incomplete optimizer update count')
        for key in ('exp_avg','exp_avg_sq'):
            require(item[key].shape==value.shape and torch.isfinite(item[key]).all(),'Invalid optimizer moment')
        require((item['exp_avg_sq']>=0).all(),'Negative squared optimizer moment')
    return dict(arm=arm,epoch=epoch,step=state['step'],selected_backbone_tensors=len(selected),
        changed_backbone_tensors=len(changed),frozen_parameters_and_all_buffers_unchanged=True,
        optimizer_states=len(ids),all_optimizer_steps_correct=True)
