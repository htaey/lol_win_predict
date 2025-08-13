from django.contrib import admin
from django.urls import path
from predictor.views import HealthView, ModelInfoView, PredictView, PredictUI

urlpatterns = [
    path('admin/', admin.site.urls),
    path('health', HealthView.as_view()),
    path('model-info', ModelInfoView.as_view()),
    path('predict', PredictView.as_view()),
    path('ui', PredictUI.as_view()),
]
