
import argparse,fcntl,hashlib,os,sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mef_vr import runtime as rt
import numpy as np

def triples(rows,seed,epochs):
    pairs={};hn=[]
    for r in rows:
        if r['role']=='hard_negative':hn.append(r)
        elif r['role'] in ('bug','clean_or_fixed'):pairs.setdefault(r['pair_id'],[]).append(r)
        else:raise ValueError('Unknown role')
    ids=sorted(pairs);hn.sort(key=lambda r:r['sample_id'])
    if len(ids)!=100 or len(hn)!=100:raise ValueError('Need all 100 Bug/Clean pairs and 100 HN')
    groups={}
    for pid,negative in zip(ids,hn):
        pair=sorted(pairs[pid],key=lambda r:r['role'])
        if [r['role'] for r in pair]!=['bug','clean_or_fixed']:raise ValueError('Broken pair')
        groups[pid]=pair+[negative]
    rng=np.random.default_rng(seed)
    return groups,[(e+1,str(pid)) for e in range(epochs) for pid in rng.permutation(ids)]

def main():
    import torch
    from flicker_component_ablation import AblatedStateHead,canonical_names
    from flicker_video_context import VideoContextRoutingHead
    from flicker_multimodel_core import supervision_loss
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True,help='Train-300 evaluate output made with --keep-features')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--head',choices=['full','context'],required=True)
    p.add_argument('--arm',choices=['full','no_dino','no_raft','no_r3d_lineage','no_c_mask'],default='full')
    p.add_argument('--seed',type=int,default=20261007);p.add_argument('--epochs',type=int,default=6)
    p.add_argument('--check-only',action='store_true',help='Validate data and feature inventory without training')
    a=p.parse_args()
    if a.head=='context' and a.arm!='full':raise ValueError('Retrained component arms here apply to Full only')
    if a.epochs<1:raise ValueError('Positive epochs required')
    rows=rt.read(ROOT/'data/labels/train.json');groups,sequence=triples(rows,a.seed,a.epochs)
    sources={r['sample_id']:r for r in rt.read(ROOT/'data/manifests/train.json')}
    job=rt.read(a.features/'job.json')
    if job['split']!='train' or set(job['ids'])!=set(sources) or not job['keep_features']:
        raise ValueError('Features must cover the entire Train-300 cohort')
    bindings={};paths={}
    for r in rows:
        sid=r['sample_id'];key=hashlib.sha256(sid.encode()).hexdigest()[:20]
        folder=a.features/'cases'/key;feature=folder/'work/features.npy';target=ROOT/r['targets']
        done=rt.read(folder/'completed.json');source=sources[sid]
        if done['source_video_sha256']!=source['video_sha256']:raise ValueError('Wrong feature video')
        if rt.read(folder/'work/features.json')['feature_names']!=canonical_names():raise ValueError('Wrong channels')
        if rt.sha(target)!=r['target_sha256']:raise ValueError('GT changed')
        x=np.load(feature,mmap_mode='r',allow_pickle=False)
        with np.load(target,allow_pickle=False) as y:
            if x.shape!=(len(y['frame']),51,*y['grid'].shape[1:]):raise ValueError('Feature/target geometry mismatch')
        paths[sid]=(feature,target);bindings[sid]=dict(features=rt.sha(feature),targets=rt.sha(target))
    if a.check_only:
        print(dict(validated_sources=len(rows),optimizer_steps=len(sequence),passed=True));return
    if a.output.exists():raise FileExistsError('Use a fresh training output directory')
    a.output.mkdir(parents=True)
    cfg=rt.config();norm=rt.npz(cfg['normalization_path'])
    rt.save(a.output/'protocol.json',dict(head=a.head,arm=a.arm,seed=a.seed,epochs=a.epochs,
        steps=len(sequence),inputs=bindings,normalization_sha256=rt.sha(cfg['normalization_path']),
        optimizer='AdamW',lr=.001,weight_decay=.0001,gradient_clip=1.,
        loss='mean of three full-video losses: balanced state + .2 boundary + .2 region',
        upstream_frozen=True,real_data_used_for_training=False,automatic_model_promotion=False))
    with Path(os.environ.get('MEF_VR_GPU_LOCK','/tmp/vrflicker_real_gt_three_models_20261014_gpu.lock')).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
        torch.set_num_threads(2);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=(AblatedStateHead(canonical_names(),a.arm) if a.head=='full' else VideoContextRoutingHead('frame_context')).cuda().train()
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        mu=torch.from_numpy(norm['mean']).cuda().float().reshape(1,51,1,1)
        sd=torch.from_numpy(norm['std']).cuda().float().reshape(1,51,1,1)
        for step,(epoch,pid) in enumerate(sequence,1):
            opt.zero_grad(set_to_none=True);total=0.
            for r in groups[pid]:
                f,t=paths[r['sample_id']]
                x=(torch.from_numpy(np.array(np.load(f,mmap_mode='r'),copy=True)).cuda().float()-mu)/sd
                with np.load(t,allow_pickle=False) as y:
                    fy=torch.from_numpy(y['frame']).cuda();gy=torch.from_numpy(y['grid']).cuda()
                loss,_=supervision_loss(model(x),fy,gy)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                (loss/3).backward();total+=float(loss.detach())/3
                del x,fy,gy,loss
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            if step%100==0 or step==len(sequence):
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),head=a.head,arm=a.arm,
                    step=step,epoch=epoch,seed=a.seed),a.output/f'checkpoint_epoch{epoch:02d}.pt')
            if step==1 or step%10==0:print(dict(step=step,total_steps=len(sequence),loss=total),flush=True)
        rt.save(a.output/'completed.json',dict(optimizer_steps=len(sequence),head=a.head,arm=a.arm,
            detector_weights_overwritten=False))

if __name__=='__main__':main()
