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

        # objective 목록: 없으면 model_meta.json에서 대체, 그래도 없으면 기본값
        try:
            _FEATURES['objectives'] = _load_json('objective_features.json')
        except FileNotFoundError:
            meta = _load_json('model_meta.json')
            _FEATURES['objectives'] = meta.get('objective_features', [
                'firstdragon','firstherald','firsttower','firstblood','firstmidtower'
            ])

        _FEATURES['meta'] = _load_json('model_meta.json')

        # 🔧 핵심: 학습 때처럼 feat + objectives 를 합친 리스트로 교정
        objs = _FEATURES.get('objectives', [])
        _FEATURES['feat10'] = list(dict.fromkeys(_FEATURES['feat10'] + objs))
        _FEATURES['feat15'] = list(dict.fromkeys(_FEATURES['feat15'] + objs))

        # (이하 모델 로드 동일)
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

    # 🔧 학습 컬럼 기준으로 DataFrame을 만들되, 없는 값은 0으로 보충
    # (오브젝트는 기본 0이 합리적, 숫자 피처도 일단 0으로; UI에서 입력하면 덮어씁니다)
    row = [feature_dict.get(c, 0) for c in cols]

    # 참고: 진짜로 누락 목록을 클라이언트에 알려주고 싶으면 아래 주석을 사용
    # missing = [c for c in cols if c not in feature_dict]
    # if missing: ... (400으로 돌려주기)

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
