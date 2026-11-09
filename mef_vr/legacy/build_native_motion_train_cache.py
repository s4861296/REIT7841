

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
import torchvision
from torchvision.models.optical_flow import raft_large
from probe_native_raft_execution import native_frames, sha, save
from extract_native_raft_train_signals import ordered_train_records
from native_dense_flow_evidence import prepare_raft_pair, native_flow_evidence
from native_flow_consistency import bidirectional_consistency
from native_motion_sequence_cache import write_sequence, MotionSequenceReader, validate_source


OBSERVED_KEYS = {'sample_id', 'video_path', 'video_sha256', 'frame_count', 'fps', 'height', 'width'}


def scan_source(record, frame_reader=native_frames):
    
    if set(record) != OBSERVED_KEYS: raise ValueError('Producer accepts observed source fields only')
    if sha(record['video_path']) != record['video_sha256']: raise ValueError('Video hash differs')
    pts = []; time_base = None
    for index, rgb, timing in frame_reader(record):
        if index != len(pts) or rgb.shape != (record['height'],record['width'],3) or rgb.dtype != np.uint8:
            raise ValueError('Source index/RGB geometry mismatch')
        if time_base is not None and timing['time_base'] != time_base: raise ValueError('Changing source time base')
        time_base = timing['time_base']; pts.append(timing['pts'])
    source = dict(source_video_sha256=record['video_sha256'],frame_count=record['frame_count'],
                  height=record['height'],width=record['width'],fps=record['fps'],pts=pts,time_base=time_base)
    validate_source(source)
    if sha(record['video_path']) != record['video_sha256']: raise ValueError('Source changed during scan')
    return source


def observed_transitions(record, source, infer_pair, frame_reader=native_frames, progress=lambda t:None):
    
    if set(record) != OBSERVED_KEYS: raise ValueError('Unexpected answer metadata')
    previous = None; seen = 0
    for index, rgb, timing in frame_reader(record):
        if (index != seen or index >= source['frame_count'] or timing['pts'] != source['pts'][index]
                or timing['time_base'] != source['time_base']
                or rgb.shape != (source['height'],source['width'],3) or rgb.dtype != np.uint8):
            raise ValueError('Decode pass changed original coordinates/PTS')
        seen += 1; progress(seen)
        if previous is not None:
            maps = infer_pair(previous, rgb)
            if not isinstance(maps, tuple) or len(maps) != 3: raise ValueError('Expected aligned/support/consistency only')
            yield index, *maps
        previous = rgb.copy()
    if seen != source['frame_count']: raise ValueError('Incomplete decoded video')
    if sha(record['video_path']) != record['video_sha256']: raise ValueError('Source changed during production')


def readback(path, source, producer):
    
    digest = hashlib.sha256(); frames = 0
    with MotionSequenceReader(path, source, producer) as reader:
        for index in range(source['frame_count']):
            cues = reader.chunk(index,index+1)
            if cues.shape != (1,4,source['height'],source['width']): raise ValueError('Native cue dimensions differ')
            if index == 0 and np.any(cues): raise ValueError('First frame fabricated')
            digest.update(cues.tobytes()); frames += 1
    return dict(source_frames=frames,cue_sha256=digest.hexdigest(),cache_sha256=sha(path),bytes=path.stat().st_size)


def cache_one(record, output_dir, index, producer, infer_pair, *, max_total_bytes, min_free_bytes,
              frame_reader=native_frames, progress=lambda t:None, expected_entry=None):
    
    output_dir = Path(output_dir)
    source = scan_source(record,frame_reader)
    prefix = f'sequence_{index:03d}_attempt'
    complete = sorted(output_dir.glob(prefix+'*.zip'))
    if len(complete)>1: raise ValueError('Ambiguous multiple committed copies; do not select silently')
    if expected_entry is not None and not complete: raise ValueError('Previously committed cache missing')
    if complete:
        if expected_entry is not None and (expected_entry['sample_id']!=record['sample_id']
                or expected_entry['source']!=source or expected_entry['path']!=str(complete[0])
                or expected_entry['cache_sha256']!=sha(complete[0])):
            raise ValueError('Previously committed inventory binding changed')
        report = readback(complete[0],source,producer)
        return dict(sample_id=record['sample_id'],source=source,path=str(complete[0]),reused=True,**report)
    attempt=0
    while (output_dir/f'{prefix}{attempt:03d}.zip.partial').exists(): attempt+=1
    path=output_dir/f'{prefix}{attempt:03d}.zip'
    used=sum(p.stat().st_size for p in output_dir.rglob('*') if p.is_file())
    remaining=max_total_bytes-used
    if remaining<=0: raise OSError('Aggregate cache quota reached; old/failed data retained')
    write_sequence(path,source,producer,observed_transitions(record,source,infer_pair,frame_reader,progress),
                   max_bytes=remaining,min_free_bytes=min_free_bytes)
    report=readback(path,source,producer)
    return dict(sample_id=record['sample_id'],source=source,path=str(path),reused=False,**report)


def make_raft_inference(weights_path):
    
    model=raft_large(weights=None,progress=False)
    model.load_state_dict(torch.load(weights_path,map_location='cpu',weights_only=True),strict=True)
    model=model.cuda().eval().requires_grad_(False)
    @torch.inference_mode()
    def infer(previous,current):
        prev=torch.from_numpy(np.ascontiguousarray(previous.transpose(2,0,1))).unsqueeze(0).cuda()
        curr=torch.from_numpy(np.ascontiguousarray(current.transpose(2,0,1))).unsqueeze(0).cuda()
        x,y,pad=prepare_raft_pair(curr,prev)
        model.corr_block.corr_pyramid=[]
        backward=pad.unpad(model(x,y,num_flow_updates=12)[-1]).contiguous()
        model.corr_block.corr_pyramid=[]
        forward=pad.unpad(model(y,x,num_flow_updates=12)[-1]).contiguous()
        model.corr_block.corr_pyramid=[]
        maps=native_flow_evidence(prev,curr,backward,forward)
        flags=bidirectional_consistency(backward,forward)
        if not torch.equal(flags['sampling_support'],maps['sampling_support']): raise ValueError('Inconsistent correspondence geometry')
        return (maps['aligned_residual'][0].cpu().numpy(),flags['sampling_support'][0].cpu().numpy(),
                flags['consistent'][0].cpu().numpy())
    return infer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('evidence','pairing-run','execution-probe','output'): parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--scope',choices=('execution_triple','train300'),required=True)
    parser.add_argument('--validated-triple',type=Path)
    parser.add_argument('--max-total-gib',type=int,default=160)
    parser.add_argument('--reserve-gib',type=int,default=40)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();out=args.output
    if not args.worker:
        if args.resume:
            code=out/'code'/Path(__file__).name
            if not code.is_file(): raise ValueError('No snapshot to resume')
        else:
            out.mkdir(parents=True,exist_ok=False);(out/'code').mkdir()
            for module in list(sys.modules.values()):
                filename=getattr(module,'__file__',None)
                if filename:
                    p=Path(filename).resolve()
                    if p.parent==Path(__file__).resolve().parent and p.suffix=='.py': shutil.copy2(p,out/'code'/p.name)
            code=out/'code'/Path(__file__).name
        os.execv(sys.executable,[sys.executable,str(code),*sys.argv[1:],'--worker'])
    lock=(out/'producer.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(pid=os.getpid(),status='preparing',started_utc=datetime.now(timezone.utc).isoformat())
    def update(**kwargs): state.update(kwargs);save(out/'status.json',state)
    update()
    try:
        if args.max_total_gib<1 or args.reserve_gib<1: raise ValueError('Positive explicit disk limits required')
        paths=[args.evidence/'protocol.json',args.pairing_run/'protocol.json',args.execution_probe/'protocol.json']
        ep,pairing,probe=[json.loads(p.read_text()) for p in paths]
        if not {'pair_order','hn_pairing'}<=set(pairing):
            raise ValueError('pairing-run must be an audited native formal run with frozen pair_order/hn_pairing')
        bindings={str(p):sha(p) for p in paths};rows=[]
        for cohort in ('train_paired','train_hn'):
            path=args.evidence/ep['cohort_manifests'][cohort]
            if sha(path)!=ep['output_sha256'][path.name]: raise ValueError('Changed Train manifest')
            bindings[str(path)]=sha(path);rows += [json.loads(line) for line in path.read_text().splitlines()]
        records=ordered_train_records(rows,pairing)
        if args.scope=='execution_triple': records=records[:3]
        weights=args.execution_probe/'raft_large_C_T_SKHT_V2-ff5fadd5.pth'
        if sha(weights)!=probe['weights_sha256']: raise ValueError('Frozen RAFT weights differ')
        bindings[str(weights)]=sha(weights)
        code_hashes={p.name:sha(p) for p in (out/'code').glob('*.py')}
        implementation=dict(code=code_hashes,torch=torch.__version__,torchvision=torchvision.__version__,
                            tf32=False,amp=False,cudnn_benchmark=False,strict_deterministic=True)
        producer=dict(algorithm='torchvision_raft_large_bidirectional',weights_sha256=probe['weights_sha256'],
                      implementation_sha256=hashlib.sha256(json.dumps(implementation,sort_keys=True).encode()).hexdigest(),
                      iterations=12,flow_dtype='float32',consistency_alpha=.01,consistency_beta=.5)
        config=dict(schema='native_motion_train_cache_v1',scope=args.scope,producer=producer,implementation=implementation,
                    inputs=bindings,source_order=[r['sample_id'] for r in records],
                    max_total_bytes=args.max_total_gib*1024**3,reserve_bytes=args.reserve_gib*1024**3,
                    detector_updates=0,test_used=False,real_gt_used=False,development_used=False,
                    all_original_frames=True,source_resize=False,first_frame_retained=True)
        protocol_path=out/'protocol.json'
        if protocol_path.exists():
            if json.loads(protocol_path.read_text())!=config: raise ValueError('Resume configuration differs')
        else: save(protocol_path,config)
        if args.scope=='train300':
            if args.validated_triple is None: raise ValueError('Require completed new-format execution triple first')
            report=json.loads((args.validated_triple/'report.json').read_text())
            if (report.get('completed') is not True or report.get('scope')!='execution_triple'
                    or report.get('producer')!=producer or len(report.get('inventory',[]))!=3
                    or report.get('input_sha256')!=bindings
                    or report.get('protocol_sha256')!=sha(args.validated_triple/'protocol.json')
                    or [r['sample_id'] for r in report['inventory']]!=[r['sample_id'] for r in records[:3]]):
                raise ValueError('Execution triple not complete with same frozen producer')
            for item in report['inventory']:
                if sha(item['path'])!=item['cache_sha256']: raise ValueError('Probe cache changed')
                readback(Path(item['path']),item['source'],producer)
        
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<12*1024**3:
            raise RuntimeError('Need 12 GiB free; do not disturb active training')
        torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.25)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True,warn_only=False)
        infer=make_raft_inference(weights)
        cache=out/'sequences';cache.mkdir(exist_ok=True);inventory=[]
        prior=json.loads((out/'inventory.json').read_text()) if (out/'inventory.json').exists() else []
        previous={r['sample_id']:r for r in prior}
        if len(previous)!=len(prior) or not set(previous)<={r['sample_id'] for r in records}:
            raise ValueError('Resume inventory outside frozen source order')
        for index,record in enumerate(records):
            update(status='producing_original_frames',video_index=index+1,video_total=len(records),sample_id=record['sample_id'])
            observed={k:record[k] for k in OBSERVED_KEYS}
            item=cache_one(observed,cache,index,producer,infer,
                max_total_bytes=config['max_total_bytes'],min_free_bytes=config['reserve_bytes'],
                progress=lambda t:update(source_frame_end=t,total_source_frames=record['frame_count']),
                expected_entry=previous.get(record['sample_id']))
            inventory.append(item);save(out/'inventory.json',inventory)
        for path,digest in bindings.items():
            if sha(path)!=digest: raise ValueError('Bound input changed during run')
        save(out/'report.json',dict(completed=True,scope=args.scope,producer=producer,inventory=inventory,
             source_frames=sum(x['source']['frame_count'] for x in inventory),optimizer_steps=0,
             directed_flow_passes_produced=sum(2*(x['source']['frame_count']-1) for x in inventory if not x['reused']),
             cached_bytes=sum(x['bytes'] for x in inventory),input_sha256=bindings,protocol_sha256=sha(protocol_path),
             efficacy_evaluated=False,test_used=False,real_gt_used=False,development_used=False))
        update(status='completed',completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        update(status='failed',error=repr(error));raise
    finally: lock.close()


if __name__=='__main__': main()
