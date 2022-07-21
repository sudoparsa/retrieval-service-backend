from rest_framework.response import Response
from rest_framework.views import APIView
from informationretrieval.clustering import kmeans_clustering_model


class ClusteringView(APIView):

    def get(self, request, *args, **kwargs):
        query = self.request.query_params['query']
        k = 10 if 'k' not in self.request.query_params else int(self.request.query_params['k'])
        k = min(k, 20)
        return Response(kmeans_clustering_model.run(query, k))


class ClusteringResultsView(APIView):

    def get(self, request, *args, **kwargs):
        return Response(kmeans_clustering_model.clustering_metrics)
