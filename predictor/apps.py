from django.apps import AppConfig

class PredictorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predictor'

    def ready(self):
        from .utils import load_artifacts_once
        load_artifacts_once()
