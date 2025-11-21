import os, json, joblib
import pandas as pd
from threading import Lock

_LOCK = Lock()
_MODELS = {}
_FEATURES = {}

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')

def _load_json(name):
    with open(os.path.join(ARTIFACT_DIR, name), 'r', encoding='utf-8') as f:
        return json.load(f)

def _load_pkl(name):
    return joblib.load(os.path.join(ARTIFACT_DIR, name))

def load_artifacts_once():
    global _MODELS, _FEATURES
    with _LOCK:
        if _MODELS:
            return

        _FEATURES['feat10'] = _load_json('feature_names_10.json')
        _FEATURES['feat15'] = _load_json('feature_names_15.json')

        try:
            _FEATURES['objectives'] = _load_json('objective_features.json')
        except FileNotFoundError:
            meta = _load_json('model_meta.json')
            _FEATURES['objectives'] = meta.get('objective_features', [
                'firstdragon','firstherald','firsttower','firstblood','firstmidtower'
            ])

        _FEATURES['meta'] = _load_json('model_meta.json')

        objs = _FEATURES.get('objectives', [])
        _FEATURES['feat10'] = list(dict.fromkeys(_FEATURES['feat10'] + objs))
        _FEATURES['feat15'] = list(dict.fromkeys(_FEATURES['feat15'] + objs))

        _MODELS['rf_10']  = _load_pkl('rf_10.pkl')
        _MODELS['xgb_10'] = _load_pkl('xgb_10.pkl')
        _MODELS['lr_10']  = _load_pkl('lr_10.pkl')
        _MODELS['rf_15']  = _load_pkl('rf_15.pkl')
        _MODELS['xgb_15'] = _load_pkl('xgb_15.pkl')
        _MODELS['lr_15']  = _load_pkl('lr_15.pkl')
        _MODELS['meta']    = _load_pkl('meta_model.pkl')
        _MODELS['meta10']  = _load_pkl('meta_model10.pkl')
        _MODELS['meta15']  = _load_pkl('meta_model15.pkl')

def required_features(which: str):
    if which == 'at10': return _FEATURES['feat10']
    if which == 'at15': return _FEATURES['feat15']
    raise ValueError("which must be 'at10' or 'at15'")

def dict_to_df(feature_dict: dict, which: str) -> pd.DataFrame:
    cols = required_features(which)
    row = [feature_dict.get(c, 0) for c in cols] # 기입안한 값 0으로 대체
    return pd.DataFrame([row], columns=cols)

def assemble_meta(prob10: dict, prob15: dict):
    meta_X = pd.DataFrame([{
        'rf_10': prob10['rf'], 'xgb_10': prob10['xgb'], 'lr_10': prob10['lr'],
        'rf_15': prob15['rf'], 'xgb_15': prob15['xgb'], 'lr_15': prob15['lr'],
    }])
    meta_10X = pd.DataFrame([{
        'rf_10': prob10['rf'], 'xgb_10': prob10['xgb'], 'lr_10': prob10['lr'],
    }])
    meta_15X = pd.DataFrame([{
        'rf_15': prob15['rf'], 'xgb_15': prob15['xgb'], 'lr_15': prob15['lr'],
    }])
    return meta_X, meta_10X, meta_15X
