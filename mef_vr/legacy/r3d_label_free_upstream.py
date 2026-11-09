
import numpy as np
import torch


def aggregate_source_windows(frame_count, starts, window_frames, predict_window):
    if type(frame_count) is not int or frame_count<1 or type(window_frames) is not int or window_frames<1:
        raise ValueError('Invalid source/window size')
    if not starts or len(set(starts))!=len(starts) or any(type(s) is not int or not 0<=s<frame_count for s in starts):
        raise ValueError('Invalid source window starts')
    total=np.zeros(frame_count,np.float64);coverage=np.zeros(frame_count,np.int32)
    for start in starts:
        probabilities=np.asarray(predict_window(start))
        if (probabilities.shape!=(window_frames,) or not np.isfinite(probabilities).all()
                or (probabilities<0).any() or (probabilities>1).any()):
            raise ValueError('Invalid complete-window probabilities')
        count=min(window_frames,frame_count-start)
        total[start:start+count]+=probabilities[:count];coverage[start:start+count]+=1
    if (coverage==0).any():raise ValueError('Not all source frames covered')
    return total/coverage,coverage


def predict_upstream_frames(common, exact, backbone, head, rgb, *, window_frames,
                            window_stride, feature_mode, difference_context):
    if (not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=4
            or rgb.shape[-1]!=3 or min(rgb.shape[:3])<1):raise ValueError('Complete native RGB required')
    if backbone.training or head.training:raise ValueError('Require explicitly loaded eval models')
    if difference_context not in ('window_zero','source_previous'):raise ValueError('Explicit original context required')
    starts=common.window_starts(len(rgb),window_frames,window_stride)
    def forward(start):
        with torch.inference_mode():
            tensor,_=common.normalized_window(rgb,start,window_frames,'cuda:0')
            features=exact.temporal_features(backbone,tensor,feature_mode)
            previous=exact.previous_source_frame(rgb,start,difference_context,'cuda:0')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if getattr(head,'requires_next_source_frame',False):
                    following=exact.following_source_frame(rgb,start,window_frames,'cuda:0')
                    raw=head(features,tensor,previous,following)
                else:raw=head(features,tensor,previous)
                probabilities=raw.float().sigmoid()[0].cpu().numpy()
        return probabilities
    return aggregate_source_windows(len(rgb),starts,window_frames,forward)
