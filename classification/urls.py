from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.index, name='home'),
    path('analyze/', views.get_prediction, name='analyze'),
]