
from fractions import Fraction
import numpy as np

VERSION='sealed_endpoint_native_array_API_v1'
RULE_ID='direct_observation_required_at_endpoint__8x8_max__D1_over255'
SELECTION_SHA256='95ab5f348b758fe993bb884f8ba651c56acf9cfdbb77edd79c8eef20090c6d52'
LAGS=(1,2,3,4,8,16)


def require(ok,message):
    if not ok:raise ValueError(message)


def near3(mask):
    out=mask.copy()
    for lag in (1,2,3):
        if len(mask)>lag:
            out[lag:]|=mask[:-lag];out[:-lag]|=mask[lag:]
    return out


def validate_clock(clock,n):
    result={k:np.asarray(clock[k]) for k in ('source_frame_indices','source_pts','source_time_s')}
    require(all(v.shape==(n,) for v in result.values()),'One clock value per original frame required')
    for key in ('source_frame_indices','source_pts'):
        require(result[key].dtype.kind in 'iu','Integer original clock required')
    require((np.diff(result['source_frame_indices'])==1).all() and (np.diff(result['source_pts'])>0).all(),
        'Dropped/repeated/reordered original frames or timestamps')
    base=Fraction(clock['time_base_fraction']);require(base>0,'Positive rational timebase required')
    expected=np.asarray([float(int(p)*base) for p in result['source_pts']],np.float64)
    require(np.array_equal(result['source_time_s'],expected),'Native PTS and seconds disagree')
    return dict(**{k:v.copy() for k,v in result.items()},time_base_fraction=str(base))


def bind_clock(obs,clock):
    for key in ('source_frame_indices','source_pts','source_time_s'):
        require(np.array_equal(np.asarray(obs[key]),clock[key]),'Observation/native clock mismatch: '+key)
    require(str(np.asarray(obs['source_time_base']).item())==clock['time_base_fraction'],'Observation timebase differs')


def predict(observation):
    
    full,context=[np.asarray(observation[k],np.float64) for k in ('full','context')]
    require(full.ndim==1 and context.shape==full.shape and len(full)>=2,'Paired native heads and at least2 frames required')
    n=len(full);clock=validate_clock(observation['clock'],n)
    for score in (full,context):
        require(np.isfinite(score).all() and ((score>=0)&(score<=1)).all(),'Invalid original head values')
    photo,interp,hist=[observation[k] for k in ('photo','interp','hist')]
    for obs in (photo,interp,hist):bind_clock(obs,clock)
    for key in ('luminance_mean','dark_fraction','bright_fraction'):
        values=np.asarray(photo[key])
        require(values.shape==(n,81) and values.dtype.kind in 'fiu' and np.isfinite(values).all()
            and ((values>=0)&(values<=1)).all(),'Invalid full-native photo statistic: '+key)
    weights=np.asarray(photo['tile_area_weights'],np.float64)
    require(weights.shape==(81,) and np.isfinite(weights).all() and (weights>0).all(),'Native positive region areas required')
    for sl in (slice(0,1),slice(1,17),slice(17,81)):
        require(np.isclose(weights[sl].sum(),1,rtol=0,atol=1e-6),'Each native grid must cover full viewport')
    known=np.asarray(interp['known']);expected=np.zeros(n,bool);expected[1:-1]=True
    require(known.dtype==np.bool_ and np.array_equal(known,expected),'RIFE first/last must remain unknown')
    residual,raw=[np.asarray(interp[k],np.float64) for k in ('residual_mean','raw_min_neighbor_mean')]
    for value in (residual,raw):
        require(value.shape==(n,81) and np.isnan(value[~known]).all(),'Unknown RIFE support cannot be zero/imputed')
        require(np.isfinite(value[known]).all() and ((value[known]>=0)&(value[known]<=1)).all(),'Invalid supported RIFE observations')
    sides=np.asarray(hist['known']);D=np.asarray(hist['D_rgb_l1'])
    expected_sides=np.zeros((n,2),bool);expected_sides[1:,0]=True;expected_sides[:-1,1]=True
    require(sides.dtype==np.bool_ and np.array_equal(sides,expected_sides),'Missing/altered one-sided native support')
    require(D.dtype==np.float64 and D.shape==(n,2,81) and np.isnan(D[~sides]).all()
        and np.isfinite(D[sides]).all() and ((D[sides]>=0)&(D[sides]<=1)).all(),'Invalid native RGB difference')
    require(np.array_equal(D[1:,0],D[:-1,1]),'The same adjacent pair disagrees between sides')
    
    
    x=np.asarray(photo['dark_fraction'],np.float64)[:,1:17]
    coverage=np.full((n,len(LAGS)),np.nan);lag_known=np.zeros((n,len(LAGS)),bool)
    for j,lag in enumerate(LAGS):
        if n<=2*lag:continue
        before,center,after=x[:-2*lag],x[lag:-lag],x[2*lag:]
        strength=np.clip(np.minimum(abs(center-before),abs(center-after))-abs(before-after),0,1)
        coverage[lag:-lag,j]=((strength>=.05)*weights[1:17]).sum(axis=1)
        lag_known[lag:-lag,j]=True
    dark_centers=(lag_known&(coverage>=.5)).any(axis=1)
    dark=bool(dark_centers.any());full_gate=bool(full.max()>=.5)
    neural=(context>=.9)&full_gate;photo_positive=bool(dark or neural.any())
    ratio=residual/(raw+1/255)
    confirm=known&(residual[:,0]>=.01)&(ratio[:,0]>=.25)
    rescue=known&(residual[:,0]>=.04)&(ratio[:,0]>=.25)
    weak_same=(full>=.25)&(context>=.5);weak_near=near3(weak_same);nearby=near3(confirm)
    composed=int(dark or (neural&(nearby|~known)).any()) if photo_positive else int((rescue&weak_near).any())
    endpoint_D=np.full(n,np.nan);endpoint_D[0]=D[0,1,17:81].max();endpoint_D[-1]=D[-1,0,17:81].max()
    direct=(~known)&(endpoint_D>=1/255)
    kept=neural&((known&nearby)|direct)
    prediction=int(dark or kept.any()) if photo_positive else composed
    require(prediction<=composed,'Endpoint rule cannot add composed positives')
    counts=dict(native_frames=n,baseline_prediction=composed,dark_rescue_preserved=dark,
        rescue_branch_preserved=not photo_positive,known_center_neural_confirmations=int((neural&known&nearby).sum()),
        endpoint_neural_alarms=int((neural&~known).sum()),endpoint_alarms_with_nearby_RIFE_evidence=int((neural&~known&nearby).sum()),
        endpoint_alarms_with_direct_observed_support=int((neural&direct).sum()),retained_neural_alarms=int(kept.sum()))
    flags=dict(dark_ABA_candidate_center=dark_centers,original_neural_alarm=neural,
        RIFE_confirmation_candidate_center=confirm,RIFE_rescue_candidate_center=rescue,
        weak_same_frame_neural_support=weak_same,weak_neural_support_within3=weak_near,
        RIFE_confirmation_within3=nearby,endpoint_direct_observed_support=direct,
        retained_neural_candidate_alarm=kept,
        active_dark_candidate_center=dark_centers&photo_positive,
        active_negative_base_rescue_center=(rescue&weak_near)&(not photo_positive))
    return dict(api_version=VERSION,decision_rule_id=RULE_ID,selection_sha256=SELECTION_SHA256,
        video_prediction=prediction,counts=counts,clock=clock,
        raw_frame_heads=dict(full=full.copy(),context=context.copy()),candidate_evidence_flags=flags,
        evidence_support=dict(RIFE_center_known=known.copy(),observed_neighbor_known=sides.copy(),dark_ABA_lag_known=lag_known),
        evidence_values=dict(endpoint_max8x8_RGB_L1=endpoint_D,dark_ABA_native_area_coverage=coverage,dark_ABA_lags=np.asarray(LAGS,np.int64)),
        video_factors=dict(full_video_max=float(full.max()),full_video_gate_accepted=full_gate,photo_baseline_positive=photo_positive,
            current_composed_prediction=composed,active_branch='photo_positive_confirmation' if photo_positive else 'photo_negative_rescue'),
        unlocalized_factors=['full_video_gate is a clip-wide max, not same-frame confirmation',
            'dark ABA is a pooled video rescue; emitted centers are mechanism support, not validated event boundaries',
            'No final dense frame labels, event boundaries, or pixel masks were audited for this postprocessor'],
        frame_predictions=None,pixel_masks=None,validated_temporal_or_spatial_accuracy=False,
        evidence_semantics='Boolean computation/evidence flags only; false is not a verified normal-frame label; unknown support stays separate',
        oldC_mask_policy='Legacy oldC output is separate; no new mask generation or resurrecting previously gated masks')
