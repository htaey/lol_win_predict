from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from predictor.views import HealthView, ModelInfoView, PredictView, PredictUI

urlpatterns = [
    path('admin/', admin.site.urls),
    path('health', HealthView.as_view()),
    path('model-info', ModelInfoView.as_view()),
    path('predict', PredictView.as_view()),
    path('ui', PredictUI.as_view()),
]

# 개발 환경에서 정적 파일 서빙
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
