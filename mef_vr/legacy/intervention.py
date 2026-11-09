
from contextlib import contextmanager

ARMS = ('full', 'no_dino', 'no_raft', 'no_r3d_lineage', 'no_c_mask')


def canonical_names():
    names = [f'dino_projection_{i}' for i in range(16)]
    for lag in (1, 4, 8):
        for direction in (-1, 1):
            names += [f'dino_cosine_{direction * lag}', f'dino_available_{direction * lag}']
    names += ['rgb_mean_r', 'rgb_mean_g', 'rgb_mean_b', 'rgb_std_r', 'rgb_std_g', 'rgb_std_b',
              'raw_previous_mean', 'raw_previous_max', 'raw_next_mean', 'raw_next_max',
              'raft_aligned_mean', 'raft_support', 'raft_consistent', 'raft_available',
              'raft_aligned_max', 'frozen_C_mask_fraction']
    names += [f'rgb_patch_difference_{d * lag}' for lag in (1, 4, 8) for d in (-1, 1)]
    return names + ['frozen_r3d_context']


def removed_channels(feature_names, arm):
    if list(feature_names) != canonical_names() or arm not in ARMS:
        raise ValueError('Exact 51-channel schema and predeclared arm required')
    if arm == 'full': return ()
    if arm == 'no_dino': return tuple(range(28))
    if arm == 'no_raft': return tuple(range(38, 43))
    if arm == 'no_c_mask': return (43,)
    return (43, 50)  


@contextmanager
def masked_normalized_input(model, indices):
    indices = tuple(indices)
    if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < 51 for i in indices):
        raise ValueError('Unique valid channel indices required')
    if not indices:
        yield
        return

    def hook(module, args):
        if len(args) != 1 or args[0].ndim != 4 or args[0].shape[1] != 51:
            raise ValueError('Expected already-normalized T,51,H,W head input')
        masked = args[0].clone()
        masked[:, list(indices)] = 0
        return (masked,)

    handle = model.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def predict_heads(models, features, mean, std, arm, *, device, infer_fn=None, baseline_context=None):
    if set(models) != {'full', 'frame_context'}:
        raise ValueError('Exactly the two original deployed heads are required')
    indices = removed_channels(canonical_names(), arm)
    if infer_fn is None:
        from flicker_retained_cuda_streaming import complete_retained_head
        infer_fn = complete_retained_head
    result = {}
    for key in ('full', 'frame_context'):
        if key == 'frame_context' and arm == 'no_raft' and baseline_context is not None:
            
            
            if tuple(getattr(models[key], 'removed_channels', ())) != indices:
                raise ValueError('Context RAFT reuse requires its original exact internal mask')
            result[key] = baseline_context
        else:
            with masked_normalized_input(models[key], indices):
                result[key] = infer_fn(models[key], features, mean, std, device=device)
    return result
