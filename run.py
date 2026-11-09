import os,sys,argparse
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
sys.dont_write_bytecode=True
from mef_vr import runtime as rt

def main():
    parser=argparse.ArgumentParser(description='MEF-VR')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--video',type=Path)
    mode.add_argument('--metrics',action='store_true')
    mode.add_argument('--train',choices=['full','context'])
    mode.add_argument('--localization',type=Path,metavar='PREDICTION_DIRECTORY')
    mode.add_argument('--worker',choices=['primary','rife'],help=argparse.SUPPRESS)
    parser.add_argument('--output',type=Path,default=Path('outputs/run'))
    parser.add_argument('--split',choices=['real97','train','development'],default='real97')
    parser.add_argument('--ids',nargs='+')
    parser.add_argument('--ablations',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--keep-features',action='store_true')
    parser.add_argument('--results',type=Path,default=rt.ROOT/'results.json')
    parser.add_argument('--features',type=Path)
    parser.add_argument('--arm',choices=['full','no_dino','no_raft','no_r3d_lineage','no_c_mask'],default='full')
    parser.add_argument('--seed',type=int,default=20261007)
    parser.add_argument('--epochs',type=int,default=6)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.worker:rt.gpu_worker(args.output,args.worker,args.ablations)
    elif args.video:rt.detect(rt.source_scope(args.video),args.output,ablations=args.ablations,keep_features=args.keep_features)
    elif args.metrics:
        from mef_vr.evaluation import metrics_file
        metrics_file(args.results,args.output)
    elif args.localization:
        from mef_vr.localization import score
        score(args.localization,args.output)
    elif args.train:
        if not args.features:parser.error('--train requires --features')
        from mef_vr.train_fusion import main as train
        sys.argv=[sys.argv[0],'--features',str(args.features),'--output',str(args.output),'--head',args.train,
            '--arm',args.arm,'--seed',str(args.seed),'--epochs',str(args.epochs)]
        if args.check_only:sys.argv.append('--check-only')
        train()
    else:
        from mef_vr.evaluation import evaluate
        evaluate(args)

if __name__=='__main__':main()
