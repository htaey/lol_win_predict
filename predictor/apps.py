from django.apps import AppConfig

class PredictorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predictor'

    def ready(self):
        # 서버 워커 프로세스마다 1회 모델 로드
        from .utils import load_artifacts_once
        load_artifacts_once()
