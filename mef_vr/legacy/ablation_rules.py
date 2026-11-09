
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np

DECISION_API_PATH = Path(__file__).resolve().with_name('decision_api.py')
DECISION_API_SHA256 = '96ff56ccbc13b7c07807a8ff40fcd0446c97c2a072726b1fc370d0f03d2ad911'
VERSION = 'frozen_native_decision_branch_counterfactuals_v1'
VARIANTS = {
    'final': '',
    'no_full_gate': '',
    'no_context_gate': '',
    'no_dark': '',
    'no_RIFE_confirmation': '',
    'no_RIFE_rescue': '',
    'no_RIFE_all': '',
    'no_endpoint_guard': '',
    'raw_full': '',
    'raw_context': '',
    'raw_twohead': '',
    'raw_photo': '',
}
THRESHOLDS = dict(full_video=.5, context_frame=.9, weak_full=.25, weak_context=.5,
                  dark_amplitude=.05, dark_coverage=.5, RIFE_confirmation_amplitude=.01,
                  RIFE_rescue_amplitude=.04, RIFE_relative_error=.25,
                  RIFE_ratio_epsilon=1/255, endpoint_direct_D=1/255,
                  nearby_original_frames=3, dark_lags=[1,2,3,4,8,16])


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_reference_api():
    require(hashlib.sha256(DECISION_API_PATH.read_bytes()).hexdigest() == DECISION_API_SHA256,
            'Frozen reference decision API changed')
    spec = importlib.util.spec_from_file_location('_sealed_native_decision_api', DECISION_API_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REFERENCE = load_reference_api()


def _branch(*, neural, weak, dark, known, nearby, rescue, direct,
            confirmation=True, rescue_enabled=True, endpoint_guard=True):
    
    photo_positive = bool(dark or neural.any())
    if confirmation:
        support = ((known & nearby) | direct) if endpoint_guard else (nearby | ~known)
    else:
        
        support = (known | direct) if endpoint_guard else np.ones_like(known)
    if photo_positive:
        return int(dark or (neural & support).any())
    return int(rescue_enabled and (rescue & REFERENCE.near3(weak)).any())


def predict_all(observation):
    
    require(isinstance(observation, dict)
            and set(observation) == {'full', 'context', 'clock', 'photo', 'interp', 'hist'},
            'Only explicit label-free observation arrays and clock are accepted')
    reference = REFERENCE.predict(observation)  
    full, context = (reference['raw_frame_heads'][k] for k in ('full', 'context'))
    flags = reference['candidate_evidence_flags']
    known = reference['evidence_support']['RIFE_center_known']
    dark = bool(flags['dark_ABA_candidate_center'].any())
    common = dict(neural=flags['original_neural_alarm'], weak=flags['weak_same_frame_neural_support'],
                  dark=dark, known=known, nearby=flags['RIFE_confirmation_within3'],
                  rescue=flags['RIFE_rescue_candidate_center'], direct=flags['endpoint_direct_observed_support'])
    full_gate = bool(full.max() >= .5)
    context_gate = bool(context.max() >= .9)
    results = {
        'final': _branch(**common),
        'no_full_gate': _branch(**dict(common, neural=context >= .9, weak=context >= .5)),
        'no_context_gate': _branch(**dict(common, neural=full >= .5, weak=full >= .25)),
        'no_dark': _branch(**dict(common, dark=False)),
        'no_RIFE_confirmation': _branch(**common, confirmation=False),
        'no_RIFE_rescue': _branch(**common, rescue_enabled=False),
        'no_RIFE_all': _branch(**common, confirmation=False, rescue_enabled=False),
        'no_endpoint_guard': _branch(**common, endpoint_guard=False),
        'raw_full': int(full_gate),
        'raw_context': int(context_gate),
        'raw_twohead': int(full_gate and context_gate),
        'raw_photo': int((full_gate and context_gate) or dark),
    }
    require(results['final'] == reference['video_prediction'], 'Original final rule parity failed')
    require(results['no_endpoint_guard'] == reference['video_factors']['current_composed_prediction'],
            'Endpoint removal does not reproduce the original composed rule')
    require(tuple(results) == tuple(VARIANTS) and all(type(v) is int and v in (0,1) for v in results.values()),
            'Fixed binary variant inventory changed')
    return results
