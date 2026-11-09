
import math
import torch
import torch.nn.functional as F


INPUT_CHANNELS = 22
FEATURE_NAMES = (
    'current_r','current_g','current_b',
    'previous_r','previous_g','previous_b',
    'following_r','following_g','following_b',
    'reference_r','reference_g','reference_b',
    'signed_reference_r','signed_reference_g','signed_reference_b',
    'previous_linf','following_linf','reference_linf',
    'frozen_r3d_logit_over8','has_previous','has_following','fps_over30',
)


def native_inputs(current, previous, following, reference, context_logits, has_previous, has_following, fps):
    
    if current.ndim != 4 or current.shape[1] != 3 or not current.is_floating_point():
        raise ValueError('Expected native floating T,3,H,W RGB')
    if any(x.shape != current.shape or x.device != current.device for x in (previous,following,reference)):
        raise ValueError('Native image shapes/devices must agree')
    t,_,h,w=current.shape
    if context_logits.shape!=(t,) or has_previous.shape!=(t,) or has_following.shape!=(t,):
        raise ValueError('One context value and endpoint flag per original frame required')
    if not math.isfinite(float(fps)) or fps<=0:raise ValueError('Invalid source FPS')
    def plane(vector):return vector.to(current)[:,None,None,None].expand(t,1,h,w)
    signed=current-reference
    return torch.cat((current,previous,following,reference,signed,
        (current-previous).abs().amax(1,keepdim=True),
        (current-following).abs().amax(1,keepdim=True),signed.abs().amax(1,keepdim=True),
        plane(context_logits/8),plane(has_previous),plane(has_following),
        current.new_full((t,1,h,w),float(fps)/30)),dim=1)


def temporal_chunks(frames,core_frames,halo=3):
    
    if frames<1 or core_frames<1 or halo<0:raise ValueError('Invalid temporal chunk size')
    for start in range(0,frames,core_frames):
        end=min(start+core_frames,frames)
        left=max(0,start-halo);right=min(frames,end+halo)
        yield left,right,start-left,end-left


class NativePixelRefiner(torch.nn.Module):
    temporal_halo=3  

    def __init__(self,use_native_skips=True):
        super().__init__()
        self.use_native_skips=bool(use_native_skips)
        def block(ci,co,stride=1,groups=4):
            return torch.nn.Sequential(torch.nn.Conv2d(ci,co,3,stride=stride,padding=1),
                                       torch.nn.GroupNorm(groups,co),torch.nn.GELU())
        self.stem=torch.nn.Sequential(block(INPUT_CHANNELS,8),block(8,8))
        self.down2=block(8,16,stride=2)
        self.down4=block(16,32,stride=2,groups=8)
        self.temporal=torch.nn.Sequential(
            torch.nn.Conv3d(32,32,(3,1,1),padding=(1,0,0),padding_mode='replicate'),torch.nn.GELU(),
            torch.nn.Conv3d(32,32,(3,1,1),padding=(2,0,0),dilation=(2,1,1),padding_mode='replicate'),torch.nn.GELU())
        self.up2=block(48,16)
        self.up1=block(24,8)
        self.classifier=torch.nn.Conv2d(8,1,1)

    def decode_features(self,features):
        
        if features.ndim!=5 or features.shape[2]!=INPUT_CHANNELS:
            raise ValueError('Expected B,T,22,H,W complete native-coordinate features')
        b,t,c,h,w=features.shape
        if min(b,t,h,w)<1:raise ValueError('Empty input')
        
        fine=self.stem(features.reshape(b*t,c,h,w))
        mid=self.down2(fine);coarse=self.down4(mid)
        ch,cw=coarse.shape[-2:]
        sequence=coarse.reshape(b,t,32,ch,cw).permute(0,2,1,3,4)
        sequence=sequence+self.temporal(sequence)
        coarse=sequence.permute(0,2,1,3,4).reshape(b*t,32,ch,cw)
        up=F.interpolate(coarse,size=mid.shape[-2:],mode='bilinear',align_corners=False)
        up=self.up2(torch.cat((up,mid if self.use_native_skips else torch.zeros_like(mid)),dim=1))
        up=F.interpolate(up,size=(h,w),mode='bilinear',align_corners=False)
        up=self.up1(torch.cat((up,fine if self.use_native_skips else torch.zeros_like(fine)),dim=1))
        return up

    def forward(self,features):
        
        decoded=self.decode_features(features)
        b,t,_,h,w=features.shape
        return self.classifier(decoded).reshape(b,t,h,w)


def video_normalized_pixel_bce(logits,targets,total_positive,total_negative):
    
    if logits.shape!=targets.shape or total_positive<0 or total_negative<0 or total_positive+total_negative<=0:
        raise ValueError('Invalid logical-video totals or target dimensions')
    if not torch.all((targets==0)|(targets==1)):
        raise ValueError('Expected exact binary native-coordinate targets')
    truth=targets.bool()
    if int(truth.sum())>total_positive or int((~truth).sum())>total_negative:
        raise ValueError('Chunk counts exceed whole-video totals')
    loss=F.binary_cross_entropy_with_logits(logits,targets.to(logits),reduction='none')
    groups=int(total_positive>0)+int(total_negative>0)
    result=logits.sum()*0
    if total_positive:result=result+loss[truth].sum()/(groups*total_positive)
    if total_negative:result=result+loss[~truth].sum()/(groups*total_negative)
    return result
