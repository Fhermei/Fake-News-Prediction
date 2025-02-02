from django.urls import path
from .views import predict_news

urlpatterns = [
    path('', predict_news, name='predict_news'),
]