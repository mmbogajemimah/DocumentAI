from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_pdf, name='upload_pdf'),
    path('search/', views.search, name='search'),
    path('summarize/', views.summarize_text, name='summarize_text'),
    path('ask/', views.answer_question, name='answer_question'),
]