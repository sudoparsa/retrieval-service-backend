from django.urls import path

from informationretrieval import views_api

urlpatterns = [
    path('article/query_retrieval/', views_api.QueryRetrievalView.as_view()),
    path('article/classification/', views_api.ClassificationView.as_view()),
    path('article/link_analyser/', views_api.LinkAnalyserView.as_view()),
    path('article/clustering/', views_api.ClusteringView.as_view()),
    path('article/clustering_result/', views_api.ClusteringResultsView.as_view()),
]
