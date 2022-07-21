from rest_framework.response import Response
from rest_framework.views import APIView

from informationretrieval.ranking.link_analyser import pagerank_result, hits_result


class LinkAnalyserView(APIView):

    def get(self, request, *args, **kwargs):
        algo = 'pagerank' if 'algorithm' not in self.request.query_params else self.request.query_params['algorithm']
        if algo == 'hits':
            return Response(hits_result)
        return Response(pagerank_result)