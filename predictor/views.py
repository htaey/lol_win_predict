from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import TemplateView

from .serializers import PredictRequestSerializer
from .utils import _MODELS, _FEATURES, dict_to_df, assemble_meta

class HealthView(APIView):
    def get(self, request):
        return Response({"status": "ok", "version": _FEATURES['meta']['version']})

class ModelInfoView(APIView):
    def get(self, request):
        return Response({
            "version": _FEATURES['meta']['version'],
            "required_features_at10": _FEATURES['feat10'],
            "required_features_at15": _FEATURES['feat15'],
            "objective_features": _FEATURES['objectives'],
        })

def _predict_one(payload: dict, return_reversal: bool):
    prob10 = prob15 = None
    result = {}

    if 'features_at10' in payload:
        X10 = dict_to_df(payload['features_at10'], 'at10')
        prob10 = {
            "rf":  float(_MODELS['rf_10'].predict_proba(X10)[:,1][0]),
            "xgb": float(_MODELS['xgb_10'].predict_proba(X10)[:,1][0]),
            "lr":  float(_MODELS['lr_10'].predict_proba(X10)[:,1][0]),
        }
        result["p10"] = prob10

    if 'features_at15' in payload:
        X15 = dict_to_df(payload['features_at15'], 'at15')
        prob15 = {
            "rf":  float(_MODELS['rf_15'].predict_proba(X15)[:,1][0]),
            "xgb": float(_MODELS['xgb_15'].predict_proba(X15)[:,1][0]),
            "lr":  float(_MODELS['lr_15'].predict_proba(X15)[:,1][0]),
        }
        result["p15"] = prob15

    if prob10 and prob15:
        meta_X, meta_10X, meta_15X = assemble_meta(prob10, prob15)
        result["meta_prob_all"] = float(_MODELS['meta'].predict_proba(meta_X)[:,1][0])
        result["meta_prob_10"]  = float(_MODELS['meta10'].predict_proba(meta_10X)[:,1][0])
        result["meta_prob_15"]  = float(_MODELS['meta15'].predict_proba(meta_15X)[:,1][0])

        if return_reversal:
            thr = _FEATURES['meta'].get("threshold_reversal", 0.5)
            result["reversal_by_xgb"] = bool((prob10["xgb"] <= thr) and (prob15["xgb"] > thr))
    elif prob10:
        import pandas as pd
        meta_10X = pd.DataFrame([{'rf_10': prob10['rf'], 'xgb_10': prob10['xgb'], 'lr_10': prob10['lr']}])
        result["meta_prob_10"] = float(_MODELS['meta10'].predict_proba(meta_10X)[:,1][0])
    elif prob15:
        import pandas as pd
        meta_15X = pd.DataFrame([{'rf_15': prob15['rf'], 'xgb_15': prob15['xgb'], 'lr_15': prob15['lr']}])
        result["meta_prob_15"] = float(_MODELS['meta15'].predict_proba(meta_15X)[:,1][0])

    return result

@method_decorator(csrf_exempt, name='dispatch')
class PredictView(APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        ser = PredictRequestSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        payload = ser.validated_data
        try:
            if payload.get('sample'):
                out = _predict_one(payload['sample'], payload['return_reversal'])
                return Response({"version": _FEATURES['meta']['version'], "results": [out]})
            else:
                results = []
                for row in payload['samples']:
                    results.append(_predict_one(row, payload['return_reversal']))
                return Response({"version": _FEATURES['meta']['version'], "results": results})
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

class PredictUI(TemplateView):
    template_name = 'predictor/ui.html'
