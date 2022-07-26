from rest_framework.response import Response
from rest_framework.views import APIView

from informationretrieval.ranking.link_analyser import pagerank_result, hits_result, hits_result_hubs


class LinkAnalyserView(APIView):

    def get(self, request, *args, **kwargs):
        algo = 'pagerank' if 'algorithm' not in self.request.query_params else self.request.query_params['algorithm']
        if algo == 'hits':
            type = 'authorities' if 'type' not in self.request.query_params else self.request.query_params['type']
            if type == 'hubs':
                return Response(hits_result_hubs)
            return Response(hits_result)
        return Response(pagerank_result)
