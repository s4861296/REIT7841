
import torch
from r3d_native_pixel_refiner import NativePixelRefiner


class NativeRestorationRefiner(NativePixelRefiner):
    def __init__(self,use_native_skips=False):
        super().__init__(use_native_skips)
        self.restoration_head=torch.nn.Conv2d(8,3,1)
        torch.nn.init.zeros_(self.restoration_head.weight)
        torch.nn.init.zeros_(self.restoration_head.bias)

    def forward_with_restoration(self,features):
        
        decoded=self.decode_features(features)
        b,t,_,h,w=features.shape
        logits=self.classifier(decoded).reshape(b,t,h,w)
        correction=self.restoration_head(decoded).reshape(b,t,3,h,w)
        return logits,correction


def masked_clean_correction(current,paired_clean,mask):
    
    if (current.ndim!=5 or current.shape[0]!=1 or current.shape[2]!=3
            or current.shape!=paired_clean.shape or min(current.shape)<1
            or mask.shape!=(current.shape[0],current.shape[1],*current.shape[-2:])
            or not current.is_floating_point() or not paired_clean.is_floating_point()
            or paired_clean.device!=current.device or mask.device!=current.device):
        raise ValueError('Aligned one-video native RGB and mask required')
    if (not torch.isfinite(current).all() or not torch.isfinite(paired_clean).all()
            or (current<0).any() or (current>1).any() or (paired_clean<0).any() or (paired_clean>1).any()
            or not torch.all((mask==0)|(mask==1))):
        raise ValueError('Finite normalized RGB and binary supervision required')
    return ((paired_clean.detach()-current.detach())*mask.detach().unsqueeze(2)).detach()


def video_normalized_restoration_l1(correction,target,mask,total_positive,total_negative):
    
    if (correction.ndim!=5 or correction.shape[0]!=1 or correction.shape[2]!=3
            or min(correction.shape)<1 or correction.shape!=target.shape
            or mask.shape!=(1,correction.shape[1],*correction.shape[-2:])
            or not correction.is_floating_point() or not target.is_floating_point()
            or target.device!=correction.device or mask.device!=correction.device):
        raise ValueError('Aligned native correction, target and mask required')
    if (type(total_positive) is not int or type(total_negative) is not int
            or min(total_positive,total_negative)<0 or total_positive+total_negative<=0
            or (total_positive+total_negative)%(correction.shape[-2]*correction.shape[-1])
            or not torch.isfinite(correction).all() or not torch.isfinite(target).all()
            or not torch.all((mask==0)|(mask==1))):
        raise ValueError('Complete-video counts and finite exact targets required')
    truth=mask.bool()
    if int(truth.sum())>total_positive or int((~truth).sum())>total_negative:
        raise ValueError('Core counts exceed full-video counts')
    if (target.detach().masked_select((~truth).unsqueeze(2).expand_as(target))!=0).any():
        raise ValueError('Normal pixels require identity correction targets')
    loss=(correction-target.detach()).abs().mean(2)
    groups=int(total_positive>0)+int(total_negative>0)
    result=correction.sum()*0
    if total_positive:result=result+loss[truth].sum()/(groups*total_positive)
    if total_negative:result=result+loss[~truth].sum()/(groups*total_negative)
    return result
