
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

import numpy as np

from flicker_multimodel_streaming import scan_observed_video
from flicker_video_prediction import output_prediction
from flicker_cpu_real_diagnostic import review_cpu, validate_sources
from flicker_single_video_budget import estimate_single_video_disk, check_disk_budget
from flicker_component_ablation import AblatedStateHead, canonical_names
from flicker_video_context import VideoContextRoutingHead
from flicker_retained_cuda_streaming import (
    predict_retained_parent, retained_head_kind, current_cuda_configuration, check_cuda_configuration,
)
from run_flicker_multimodel_pilot import sha, utc

HERE=Path(__file__).resolve().parent
NAME=Path(__file__).name
TEST='test_flicker_retained_parent_cuda_entry.py'
VERSION='single_original_mp4_explicit_retained_cuda_v1'
GPU_PARENT=Path('/mnt/e/VRFlickerLab/model_evaluation/multimodel_historical_real_21_20261007_v1')
GPU_PARENT_SHA='18232016eeb0ff6ec48b5e931cdb56b5b9c087d8aa10274eea9b12dea5339ae8'
FRAME=Path('/mnt/e/VRFlickerLab/model_training/multimodel_video_context_full_20261010_v1')
FRAME_PROTOCOL_SHA='9c248fd071e7a43c448e3b325e0287b0bc54dae3b182cb5adef755b1876e1cc4'
FRAME600_SHA='d563decb0b70b294f339b90f81d06232e70482ca82fd37068082725d44ac684f'
GATE=Path('/mnt/e/VRFlickerLab/model_training/multimodel_component_ablation_20261009_v2')
GATE_PROTOCOL_SHA='bb2d4db2ba8c7be44729d0a04f324e885370bde8b61c402ed52e0f15d2be61dd'
GATE_SHAS={20261007:'e4c65ee8274c128d1a29c3526cb8e39a1d11d756788b782d3c646b7fc8b74f95',
           20261009:'8cdfb9609c28cf518cc30b8412a1bbe6c912290529363a66f86cc42319776b86'}
FRESH=Path('/mnt/e/VRFlickerLab/model_training/temporal_source_extension_fresh666_20261011_v1')
FRESH_PROTOCOL_SHA='61b260cb9d2e357ed2e5103f29fb08786ac92d449ede7545d8c33f6dac12246b'
FRESH_COMPLETE_SHA='54c6181771e313f8bf05aa7cab902ec3749405685925dc7d158e594047433cf6'
FRESH_REPORT_SHA='b340b28ad1f1d81239fea61b73be5073c33ef696cb44b29be86babd6ecb96a6c'
NORM_SHA='4843fbed8a6b5ae7be3e552043e9f07ab9b112b6cde87bb50c658139b3cd61ab'
STREAMING_SHA='b27644c69f853119961e2f70dc72215b34ef6854077273fbfd3869302b0330d6'
CUDA_EVIDENCE=Path('/mnt/e/VRFlickerLab/model_evaluation/temporal_source_extension_fresh666_cuda_equivalence_20261011_v1/report.json')
CUDA_EVIDENCE_SHA='969d6851213c9058b2eb2de173b54aa63a9a32822f8cf70cb27d0227a51985a9'
HEAD_DEPENDENCIES=('flicker_component_ablation.py','flicker_multimodel_core.py','flicker_video_context.py',
    'flicker_context_routing.py','flicker_temporal_connectivity.py','flicker_retained_parent_inference.py',
    'flicker_retained_cuda_streaming.py')
CONFIG=dict(upstream_window=32,upstream_stride=8,upstream_difference_context='window_zero',
    upstream_feature_mode='appearance_delta_mean_max',upstream_dense_difference_branch='shallow_native',
    upstream_head_channels=64,feature_core=32,feature_halo=8)
CLOCK_KEYS=('frame_count','height','width','fps_fraction','time_base_fraction','source_pts','source_time_s')
OBS_FILES=('source/observation.json','source/status.json','source/rgb.npy','upstream.json',
           'native_C.npz','motion_pooled.npy','motion_pooled.json','features.npy','features.json')
RESULT_FILES=('prediction.json','predicted_masks.npz','temporal_predictions.npz','review.mp4','review.json','renderer_receipt.json')
STAGE_PRECISION={
    'H':'Original frozen R3D/H BF16 autocast; cudnn TF32 enabled and deterministic algorithms disabled by original producer',
    'C':'Original native C FP32; cudnn TF32 disabled and deterministic algorithms enabled',
    'RAFT':'Original native bidirectional RAFT FP32,12 updates; deterministic algorithms enabled',
    'DINO':'Original DINO BF16 autocast then FP16 stored51 features; deterministic algorithms disabled',
    'temporal':'CPU FP32 frozen normalization BEFORE transfer; CUDA FP32, AMP/TF32 off, deterministic,32core/6halo',
}


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def publish(path,value):
    
    path=Path(path);tmp=path.with_name(path.name+'.partial_'+uuid.uuid4().hex)
    with tmp.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.link(tmp,path);tmp.unlink()


def verify(bindings):
    for path,digest in bindings.items():
        require(Path(path).is_file() and not Path(path).is_symlink() and sha(path)==digest,'Bound file changed: '+str(path))


def selection(policy='retained600',seed=20261007,arm=None):
    require(type(seed) is int and seed in GATE_SHAS,'Explicit supported seed required')
    require(policy in ('retained600','experimental_fresh666'),'No best/auto policy selection')
    if policy=='retained600':
        require(seed==20261007 and arm is None,'Retained default is final600 seed07 only; no implicit arm/seed substitution')
        return dict(policy=policy,seed=seed,model_key='frame_context',steps=600,experimental=False,promoted=True)
    require(arm in ('control_replay','candidate_extension'),'Experimental fresh666 requires explicit control/candidate arm')
    return dict(policy=policy,seed=seed,model_key=arm,steps=666,experimental=True,promoted=False)


def source_descriptor(video):
    path=Path(video).resolve(strict=True)
    require(path.is_file() and path.suffix.lower()=='.mp4','One original MP4 file required')
    digest=sha(path);source=dict(sample_id='video_'+digest[:20],video_path=str(path),video_sha256=digest,approved_seconds=None)
    validate_sources([source]);return source


def gather_models(chosen):
    
    require(chosen==selection(chosen['policy'],chosen['seed'],chosen['model_key'] if chosen['experimental'] else None),'Changed explicit selection')
    provenance={}
    def bound(path,expected=None):
        path=Path(path);digest=sha(path);require(expected is None or digest==expected,'Direct protocol/receipt changed')
        provenance[str(path)]=digest;return read(path)
    gpu=bound(GPU_PARENT/'protocol.json',GPU_PARENT_SHA)
    frame=bound(FRAME/'protocol.json',FRAME_PROTOCOL_SHA);gate=bound(GATE/'protocol.json',GATE_PROTOCOL_SHA)
    require(all(gpu[k]==v for k,v in CONFIG.items()) and gpu['raft_producer']['iterations']==12,'GPU producer configuration changed')
    require(set(gpu['models'])=={'backbone_selection','backbone','head','native_C','raft','dino'},'Frozen observer model set changed')
    norm=str(Path(frame['base'])/'normalization.npz')
    require(frame['source_bindings'][norm]==gpu['normalization_sha256']==NORM_SHA
            and gate['parent_bindings'][str(Path(gate['parent'])/'normalization.npz')]==NORM_SHA
            and gate['feature_names']==canonical_names(),'Different frozen normalization/feature schema')
    seed=chosen['seed'];heads={}
    complete=bound(GATE/f'seed_{seed}/training_complete.json')
    require(complete['completed'] and complete['checkpoint_sha256']==GATE_SHAS[seed]
            and complete['protocol_sha256']==GATE_PROTOCOL_SHA and complete.get('steps',complete.get('steps_per_arm'))==600,'Full gate is not same-seed final600')
    heads['full']=dict(path=str(GATE/f'seed_{seed}/latest_checkpoint.pt'),sha256=GATE_SHAS[seed],
        protocol_sha256=GATE_PROTOCOL_SHA,model_key='full',steps=600,seed=seed)
    if not chosen['experimental']:
        complete=bound(FRAME/f'seed_{seed}/training_complete.json')
        require(complete['completed'] and complete['checkpoint_sha256']==FRAME600_SHA
            and complete['protocol_sha256']==FRAME_PROTOCOL_SHA and complete.get('steps',complete.get('steps_per_arm'))==600,'Retained frame head is not final600')
        heads['frame_context']=dict(path=str(FRAME/f'seed_{seed}/checkpoint_epoch06.pt'),sha256=FRAME600_SHA,
            protocol_sha256=FRAME_PROTOCOL_SHA,model_key='frame_context',steps=600,seed=seed)
    else:
        complete=bound(FRESH/'completed.json',FRESH_COMPLETE_SHA);report=bound(FRESH/'report.json',FRESH_REPORT_SHA)
        fresh=bound(FRESH/'protocol.json',FRESH_PROTOCOL_SHA)
        require(complete['completed'] and report['completed'] and complete['report_sha256']==FRESH_REPORT_SHA
            and complete['protocol_sha256']==report['protocol_sha256']==FRESH_PROTOCOL_SHA
            and report['both_seeds_pass'] is False and report['automatic_promotion'] is False
            and fresh['normalization_sha256']==NORM_SHA,'Completed UNPROMOTED fresh666 experiment required')
        
        for s in GATE_SHAS:
            folder=FRESH/f'seed_{s}';tc=bound(folder/'training_complete.json');pc=bound(folder/'predictions_complete.json')
            require(tc['completed'] and pc['completed'] and tc['steps']==666 and pc['sources']==522
                and tc['protocol_sha256']==pc['protocol_sha256']==FRESH_PROTOCOL_SHA
                and tc['checkpoint_sha256']==pc['checkpoint_sha256'],'Both completed666/522 commits required')
            if s==seed:
                head=bound(folder/'head.json');receipt=bound(head['receipt_path'],head['receipt_sha256'])
                require(head['step']==666 and receipt=={k:v for k,v in head.items() if k not in ('receipt_path','receipt_sha256')}
                    and head['checkpoint_sha256']==tc['checkpoint_sha256']==report['source_bindings'][head['checkpoint_path']]
                    and Path(head['checkpoint_path']).parent==folder,'Immutable final666 checkpoint differs')
                heads['frame_context']=dict(path=head['checkpoint_path'],sha256=head['checkpoint_sha256'],
                    protocol_sha256=FRESH_PROTOCOL_SHA,model_key=chosen['model_key'],steps=666,seed=seed)
    bindings=dict(gpu['model_sha256']);bindings[norm]=NORM_SHA
    bindings.update({v['path']:v['sha256'] for v in heads.values()})
    verify(bindings)
    evidence=bound(CUDA_EVIDENCE,CUDA_EVIDENCE_SHA)
    require(evidence['completed'] and evidence['passed'] and evidence['new_MP4_end_to_end_test'] is False,
        'Completed temporal-only engineering evidence required; not new-MP4 efficacy')
    expected_head_code={}
    for name in HEAD_DEPENDENCIES:
        values={digest for path,digest in evidence['input_bindings'].items() if Path(path).name==name}
        require(len(values)==1,'Missing/conflicting validated temporal dependency: '+name)
        expected_head_code[name]=values.pop()
    require(expected_head_code['flicker_retained_cuda_streaming.py']==STREAMING_SHA,'Streaming version changed')
    for protocol in (frame,gate):
        for name,digest in protocol['code_sha256'].items():
            if name in expected_head_code:require(expected_head_code[name]==digest,'Old model dependency conflicts: '+name)
    verify({str(HERE/n):d for n,d in expected_head_code.items()})
    return dict(models=gpu['models'],model_bindings=bindings,heads=heads,normalization_path=norm,
        normalization_sha256=NORM_SHA,provenance_bindings=provenance,raft_producer=gpu['raft_producer'],
        GPU_producer_code_sha256=gpu['code_sha256'],head_code_sha256=expected_head_code,selection=chosen,**CONFIG)


def freeze_sources(files,gpu_bindings,gpu_code,head_bindings):
    
    files=dict(files);provenance={};protected={NAME,*head_bindings}
    for name,digest in gpu_bindings.items():
        require(Path(name).name==name and name.endswith('.py'),'Only flat Python GPU dependencies allowed')
        original=files.get(name);path=Path(gpu_code)/name;verify({str(path):digest})
        old_sha=sha(original) if original is not None else None
        if name in protected:
            require(original is not None and old_sha==digest,'GPU dependency conflicts with protected entry/temporal code: '+name)
        if name in head_bindings:require(digest==head_bindings[name],'GPU/model dependency SHA conflict: '+name)
        files[name]=path
        provenance[name]=dict(source_path=str(path.resolve()),source_sha256=digest,
            selection_reason='Original frozen GPU producer implementation',
            displaced_worktree_path=str(Path(original).resolve()) if original is not None else None,
            displaced_worktree_sha256=old_sha,content_changed_from_worktree=old_sha is not None and old_sha!=digest)
    for name,path in files.items():
        require(Path(name).name==name and name.endswith('.py'),'Only flat Python dependencies allowed')
        digest=sha(path)
        if name in head_bindings:require(digest==head_bindings[name],'Frozen head dependency mismatch: '+name)
        if name not in provenance:provenance[name]=dict(source_path=str(Path(path).resolve()),source_sha256=digest,
            selection_reason='New thin entry or its unchanged bound helper',displaced_worktree_path=None,
            displaced_worktree_sha256=None,content_changed_from_worktree=False)
    require(set(head_bindings)<=set(files),'Temporal code closure incomplete')
    return files,provenance


def disk_budget(clock,path):
    estimate=estimate_single_video_disk(clock['frame_count'],clock['height'],clock['width'],
        upstream_window=CONFIG['upstream_window'],upstream_stride=CONFIG['upstream_stride'])
    check_disk_budget(path,estimate)
    return dict(estimate=estimate,conservative_CPU_storage_allowance_reused=True,
        whole_pipeline_RAM_bounded=False,arbitrary_4K_or_length_guaranteed=False,
        limitation='GPU stages use mmap and temporal chunks, but native RAFT single-pair memory and packed C masks remain; no downscale/fallback')


def prepare(video,output,policy='retained600',seed=20261007,arm=None):
    out=Path(output).absolute();require(not out.exists() and not out.is_symlink() and out.parent.is_dir(),'Fresh output under existing parent required')
    chosen=selection(policy,seed,arm);source=source_descriptor(video);clock=scan_observed_video(source['video_path'])
    verify({source['video_path']:source['video_sha256']});estimate=disk_budget(clock,out.parent);p=gather_models(chosen)
    work=Path(tempfile.mkdtemp(prefix=out.name+'.partial_',dir=out.parent));(work/'code').mkdir()
    from run_r3d_audited_native_transfer import python_closure
    files=python_closure(HERE,NAME)
    
    
    files,origins=freeze_sources(files,p['GPU_producer_code_sha256'],GPU_PARENT/'code',p['head_code_sha256'])
    for name,path in files.items():shutil.copy2(path,work/'code'/name)
    verify({str(work/'code'/n):s for n,s in p['head_code_sha256'].items()})
    test=HERE.parent/'tests'/TEST;shutil.copy2(test,work/TEST)
    publish(work/'source.json',source)
    p.update(version=VERSION,created_at=utc(),source_sha256=sha(work/'source.json'),original_clock=clock,
        resource_estimate=estimate,feature_names=canonical_names(),device='cuda:0',threads=2,
        minimum_free_gpu_bytes=20*1024**3,minimum_free_disk_bytes=40*1024**3,cuda_memory_fraction=.65,
        threshold=.5,temporal_core=32,temporal_halo=6,stage_precision=STAGE_PRECISION,
        all_original_frames=True,source_resize=False,fps_resampled=False,GT_or_Clean_in_forward=False,
        annotation_manifest_loaded=False,training_updates=0,automatic_promotion=False,
        backend_numeric_equivalence_claimed=False,real_world_performance_validated=False,
        code_source_provenance=origins,
        frozen_GPU_producer_overrides_worktree_dependencies=sorted(n for n,v in origins.items() if v['content_changed_from_worktree']),
        code_sha256={n:sha(work/'code'/n) for n in files},test_sha256={TEST:sha(work/TEST)},
        limitations=['Single-seed explicit selection, not an ensemble or independently selected best model.',
            'Experimental fresh666 failed the scientific promotion gate; explicit inspection does not promote it.',
            'Old C provides spatial pixels. A time-positive frame may have no predicted pixels; no mask veto.',
            'GPU observers retain original H BF16/TF32 settings; only final temporal heads are strict FP32/noTF32.',
            'All original VFR timestamps preserved; learned frame lags are not proven FPS invariant.',
            'Review is lossy visualization; packed masks are authoritative; no root-cause classifier in this entry.'])
    publish(work/'protocol.json',p)
    env=dict(os.environ,PYTHONPATH=str(work/'code'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
    checked=subprocess.run([sys.executable,'-m','unittest',TEST[:-3],'-v'],cwd=work,env=env,text=True,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    publish(work/'tests.json',dict(returncode=checked.returncode,output=checked.stdout,CUDA_test_run=False))
    require(checked.returncode==0,'Frozen CPU tests failed; preserve partial: '+str(work))
    publish(work/'prepared.json',dict(completed=True,protocol_sha256=sha(work/'protocol.json'),tests_sha256=sha(work/'tests.json'),GPU_started=False))
    require(not out.exists(),'Concurrent output appeared; partial preserved');work.rename(out)
    return dict(prepared=str(out),selection=chosen,production_started=False)


def validate_prepared(out):
    marker=read(out/'prepared.json');p=read(out/'protocol.json')
    require(marker['completed'] and marker['protocol_sha256']==sha(out/'protocol.json')
        and marker['tests_sha256']==sha(out/'tests.json') and read(out/'tests.json')['returncode']==0,'Frozen preparation/tests changed')
    require(p['version']==VERSION and p['selection']==selection(p['selection']['policy'],p['selection']['seed'],
        p['selection']['model_key'] if p['selection']['experimental'] else None)
        and p['device']=='cuda:0' and p['threshold']==.5 and p['temporal_core']==32 and p['temporal_halo']==6
        and p['stage_precision']==STAGE_PRECISION and all(p[k]==v for k,v in CONFIG.items()),'Unsupported prepared execution policy')
    verify({str(out/'code'/n):s for n,s in p['code_sha256'].items()});verify({str(out/n):s for n,s in p['test_sha256'].items()})
    require(set(p['code_source_provenance'])==set(p['code_sha256']) and all(
        p['code_source_provenance'][n]['source_sha256']==s for n,s in p['code_sha256'].items()),'Code provenance differs from copied bytes')
    verify(p['model_bindings']);verify(p['provenance_bindings']);verify({str(out/'source.json'):p['source_sha256']})
    source=read(out/'source.json');validate_sources([source]);require(source['approved_seconds'] is None,'Complete single source only')
    verify({source['video_path']:source['video_sha256']});return p,source


def begin_run(out,resume):
    marker=out/'run_started.json'
    if marker.exists():
        require(resume and read(marker)['protocol_sha256']==sha(out/'protocol.json'),'Existing run requires explicit matching --resume')
    else:
        require(not resume and not (out/'observations').exists() and not (out/'result').exists(),'Unowned run/cache preserved; fresh output required')
        publish(marker,dict(protocol_sha256=sha(out/'protocol.json'),started_at=utc(),pid=os.getpid()))


def configure_observation_cuda(p):
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') in (None,':4096:8'),'Conflicting CUDA workspace config')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    import cv2
    require(torch.cuda.is_available(),'cuda:0 unavailable; no CPU fallback')
    require(torch.cuda.mem_get_info(0)[0]>=p['minimum_free_gpu_bytes'],'Insufficient free GPU; no eviction/downscale/fallback')
    torch.set_num_threads(p['threads']);cv2.setNumThreads(1);torch.cuda.set_per_process_memory_fraction(p['cuda_memory_fraction'])
    torch.manual_seed(20261011);torch.cuda.manual_seed_all(20261011)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False


def configure_temporal_cuda():
    import torch
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','CUDA workspace must be configured before observations')
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    return check_cuda_configuration(current_cuda_configuration())


def validate_observation(obs,source,clock,rgb,*,scope=None):
    if scope is not None:
        from flicker_approved_source_scope import validate_scoped_observation
        validate_scoped_observation(obs,source,scope,clock,rgb)
        return
    require(all(obs[k]==clock[k] for k in CLOCK_KEYS) and obs['fps']==clock['fps'],'Original clock/geometry changed')
    require(obs['video_path']==source['video_path'] and obs['video_sha256']==source['video_sha256']
        and obs['approved_seconds'] is None and obs['source_frame_count']==obs['frame_count']
        and obs['source_start_frame']==0 and obs['source_end_frame_exclusive']==obs['frame_count'],'Source/range changed')
    require(rgb.dtype==np.uint8 and rgb.shape==(obs['frame_count'],obs['height'],obs['width'],3),'Original RGB shape changed')
    require(all(obs[k] is False for k in ('spatial_resize','temporal_sampling','fps_resampled','video_reencoded','labels_loaded','paired_clean_used')),'Forbidden input transformation/annotation')


def observation_receipt(out,p,source):
    folder=out/'observations';receipt=read(folder/'completed.json')
    require(not folder.is_symlink() and receipt['completed'] and receipt['protocol_sha256']==sha(out/'protocol.json')
        and receipt['source_sha256']==p['source_sha256'] and receipt['video_sha256']==source['video_sha256']
        and set(receipt['files_sha256'])==set(OBS_FILES),'Incomplete/foreign GPU observation cache')
    require(receipt.get('source_scope')==p.get('source_scope'), 'Observation scope changed')
    verify({str(folder/n):d for n,d in receipt['files_sha256'].items()})
    require(read(folder/'source/status.json')['status']=='completed','Native decode incomplete')
    return receipt


def observe_gpu(out,p,source,update):
    import run_flicker_multimodel_observed_inference as components
    folder=out/'observations'
    if folder.exists():
        require((folder/'completed.json').is_file(),'Incomplete observation attempt preserved; use fresh output, never partial stage-cache reuse')
        observation_receipt(out,p,source)
    else:
        folder.mkdir();publish(folder/'source_started.json',dict(protocol_sha256=sha(out/'protocol.json'),source_sha256=p['source_sha256']))
        scope=p.get('source_scope')
        require(scope is not None or source['approved_seconds'] is None, 'Approved interval needs bound source scope')
        obs=components.create_observed_store(source['video_path'],folder/'source',source['video_sha256'],source['approved_seconds'])
        rgb=np.load(folder/'source/rgb.npy',mmap_mode='r',allow_pickle=False)
        validate_observation(obs,source,p['original_clock'],rgb,scope=scope)
        config={k:p[k] for k in (*CONFIG,'models','raft_producer')}
        upstream=components.upstream(folder,config,rgb,update)
        packed,maxima=components.native_mask(folder,config,rgb,obs,np.asarray(upstream['frame_probabilities'],np.float64),update)
        pooled=components.motion(folder,config,rgb,update)
        features=components.features(folder,config,rgb,np.asarray(upstream['frame_probabilities'],np.float64),packed,pooled,update)
        
        from prepare_flicker_new_train_gpu_observations import validate_outputs
        validate_outputs(folder,dict(clock=p['original_clock']),upstream,packed,maxima,pooled,features)
        verify({source['video_path']:source['video_sha256']})
        receipt=dict(completed=True,protocol_sha256=sha(out/'protocol.json'),source_sha256=p['source_sha256'],
            video_sha256=source['video_sha256'],original_frames=obs['frame_count'],labels_used=False,
            paired_clean_used=False,files_sha256={n:sha(folder/n) for n in OBS_FILES})
        if scope is not None:receipt['source_scope']=scope
        publish(folder/'completed.json',receipt)
    obs=read(folder/'source/observation.json');rgb=np.load(folder/'source/rgb.npy',mmap_mode='r',allow_pickle=False)
    validate_observation(obs,source,p['original_clock'],rgb,scope=p.get('source_scope'))
    require(read(folder/'features.json')['feature_names']==p['feature_names'],'Feature schema changed')
    with np.load(folder/'native_C.npz',allow_pickle=False) as z:packed=z['packed_masks'].copy()
    features=np.load(folder/'features.npy',mmap_mode='r',allow_pickle=False)
    return obs,rgb,packed,features


def load_heads_cpu(p):
    
    import torch
    result={}
    for kind in ('full','frame_context'):
        b=p['heads'][kind];verify({b['path']:b['sha256']})
        require(p['model_bindings'][b['path']]==b['sha256'] and b['seed']==p['selection']['seed']
            and b['model_key']==('full' if kind=='full' else p['selection']['model_key'])
            and b['steps']==(600 if kind=='full' else p['selection']['steps']),'Explicit checkpoint/seed/arm selection differs')
        state=torch.load(b['path'],map_location='cpu',weights_only=True)
        require(state['seed']==b['seed'] and state['protocol_sha256']==b['protocol_sha256']
            and len(state['history'])==b['steps'],'Checkpoint is not the selected final training step')
        model=AblatedStateHead(canonical_names(),'full') if kind=='full' else VideoContextRoutingHead('frame_context')
        model.load_state_dict(state['models'][b['model_key']],strict=True);model.eval().requires_grad_(False)
        require(retained_head_kind(model,device='cpu')==kind,'Different model architecture');result[kind]=model
    return result


def selected_output(obs,p,prediction,packed):
    result,masks=output_prediction(obs,prediction['frame_probabilities'],packed)
    result.update(backend='cuda:0',selection=p['selection'],stage_precision=p['stage_precision'],
        temporal_strategy='full_gate_frame_context',ensemble=False,automatic_promotion=False,
        video_gate_accepted=prediction['video_accepted'],full_video_max_frame_probability=prediction['full_video_max_frame_probability'],
        boundary_used_for_decoding=False,empty_output_spatial_mask_veto=False,
        frozen_C_mask_fraction_is_input_feature=True,
        spatial_method='Original frozen C native pixels gated by the explicitly selected same-seed full+frame_context policy; no new pixel decoder')
    return result,masks


def finalize_review(work,rendered,p,binding):
    
    work=Path(work);rendered=Path(rendered);metadata=rendered.with_suffix('.json');raw=read(metadata)
    require(rendered.parent==work and metadata.parent==work and raw['binding']==binding
        and raw['sha256']==sha(rendered) and raw['GT_overlay'] is False,'Renderer artifact/binding changed')
    require(not any((work/n).exists() for n in ('review.mp4','review.json','renderer_receipt.json')),'Review destinations must be fresh')
    rendered.rename(work/'review.mp4');metadata.rename(work/'renderer_receipt.json')
    result=dict(raw,selection=p['selection'],prediction_backend='cuda:0',renderer_device='cpu',
        overlay_semantics='Original frozen C predicted native pixels gated by explicitly selected full+frame_context time; not GT',
        temporal_strategy='full_gate_frame_context',automatic_promotion=False,
        stage_precision=p['stage_precision'],original_renderer_receipt_sha256=sha(work/'renderer_receipt.json'))
    publish(work/'review.json',result)
    return result


def commit_result(out,p,source):
    verify({source['video_path']:source['video_sha256']});folder=out/'result';receipt=read(folder/'receipt.json')
    require(not folder.is_symlink() and receipt['protocol_sha256']==sha(out/'protocol.json')
        and receipt['source_sha256']==p['source_sha256'] and receipt['selection']==p['selection']
        and set(receipt['artifacts'])==set(RESULT_FILES),'Foreign/incomplete result')
    verify({str(folder/n):s for n,s in receipt['artifacts'].items()})
    observation_receipt(out,p,source)
    require(receipt['observations_complete_sha256']==sha(out/'observations/completed.json'),'Consumed observation receipt changed')
    complete=dict(completed=True,receipt_sha256=sha(folder/'receipt.json'),selection=p['selection'],
        prediction='result/prediction.json',mask='result/predicted_masks.npz',review='result/review.mp4',
        temporal_heads='result/temporal_predictions.npz',original_frames=receipt['original_frames'],
        original_video_unchanged=True,labels_used=False,model_performance_claimed=False)
    if p.get('source_scope') is not None:
        complete.update(source_scope=p['source_scope'],input_scope='explicitly_approved_source_interval',
            full_file_model_context_used=False)
    if (out/'result.json').exists():require(read(out/'result.json')==complete,'Completed result pointer changed')
    else:publish(out/'result.json',complete)
    return complete


def run(output,resume=False):
    out=Path(output).resolve();require(HERE==out/'code','Run the prepared immutable code entry')
    lock=(out/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);began=False
    def update(**values):
        if values.get('status')!='failed':require(shutil.disk_usage(out).free>=40*1024**3,'Keep40GiB; preserve outputs without deleting files')
        temporary=out/('status.partial_'+uuid.uuid4().hex);publish(temporary,dict(pid=os.getpid(),updated_at=utc(),**values));os.replace(temporary,out/'status.json')
    try:
        p,source=validate_prepared(out);begin_run(out,resume);began=True
        for name in ('observations','result'):require(not (out/name).is_symlink(),'Private output directory cannot be a symlink')
        if (out/'result').exists():return commit_result(out,p,source)
        disk_budget(p['original_clock'],out);configure_observation_cuda(p)
        obs,rgb,packed,features=observe_gpu(out,p,source,update)
        actual_configuration=configure_temporal_cuda()
        models={name:model.to('cuda:0') for name,model in load_heads_cpu(p).items()}
        with np.load(p['normalization_path'],allow_pickle=False) as z:mean,std=z['mean'].copy(),z['std'].copy()
        update(status='predicting_explicit_selected_heads',selection=p['selection'],original_frames=obs['frame_count'])
        prediction=predict_retained_parent(models['full'],models['frame_context'],features,mean,std,device='cuda:0',
            source_pts=obs['source_pts'],source_time_s=obs['source_time_s'],time_base_fraction=obs['time_base_fraction'])
        result,masks=selected_output(obs,p,prediction,packed);work=Path(tempfile.mkdtemp(prefix='result.partial_',dir=out))
        np.savez_compressed(work/'predicted_masks.npz',packed_masks=masks,native_shape=np.asarray(rgb.shape[:3],np.int64),
            source_pts=np.asarray(obs['source_pts'],np.int64),source_time_s=np.asarray(obs['source_time_s'],np.float64))
        arrays={kind+'_'+key:value for kind in ('full','frame_context') for key,value in prediction[kind].items()}
        np.savez_compressed(work/'temporal_predictions.npz',**arrays,frame_probabilities=prediction['frame_probabilities'],
            frame_positive=prediction['frame_positive'],source_pts=np.asarray(obs['source_pts'],np.int64),source_time_s=prediction['source_time_s'])
        result.update(protocol_sha256=sha(out/'protocol.json'),heads=p['heads'],CUDA_configuration=actual_configuration,
            mask_artifact=dict(path='predicted_masks.npz',sha256=sha(work/'predicted_masks.npz'),bitorder='little'),
            temporal_artifact=dict(path='temporal_predictions.npz',sha256=sha(work/'temporal_predictions.npz')))
        publish(work/'prediction.json',result);update(status='rendering_original_speed_review')
        binding=dict(protocol_sha256=sha(out/'protocol.json'),prediction_sha256=sha(work/'prediction.json'))
        rendered=review_cpu(work,obs,rgb,packed,prediction['frame_probabilities'],binding)
        finalize_review(work,rendered,p,binding)
        verify(p['model_bindings']);verify({source['video_path']:source['video_sha256']})
        publish(work/'receipt.json',dict(protocol_sha256=sha(out/'protocol.json'),source_sha256=p['source_sha256'],selection=p['selection'],
            observations_complete_sha256=sha(out/'observations/completed.json'),original_frames=obs['frame_count'],
            artifacts={n:sha(work/n) for n in RESULT_FILES}))
        require(not (out/'result').exists(),'Result appeared; preserve partial');work.rename(out/'result')
        done=commit_result(out,p,source);update(status='completed',original_frames=obs['frame_count'],labels_used=False);return done
    except BaseException as error:
        if began and not (out/'result.json').exists():update(status='failed',error=repr(error))
        raise
    finally:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('prepare','run'),required=True);parser.add_argument('--video',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--policy',choices=('retained600','experimental_fresh666'),default=None)
    parser.add_argument('--seed',type=int,default=None);parser.add_argument('--arm',choices=('control_replay','candidate_extension'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.action=='prepare':
        require(args.video is not None and not args.resume,'Prepare needs one video and forbids resume')
        result=prepare(args.video,args.output,args.policy or 'retained600',20261007 if args.seed is None else args.seed,args.arm)
    else:
        require(args.video is None and args.policy is None and args.seed is None and args.arm is None,'Run uses only frozen source/model selection')
        result=run(args.output,args.resume)
    print(json.dumps(result),flush=True)
