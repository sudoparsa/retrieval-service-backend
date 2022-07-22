from rest_framework.response import Response
from rest_framework.views import APIView
from informationretrieval.retrieval_systems import *


class QueryRetrievalView(APIView):

    def get(self, request, *args, **kwargs):
        method = self.request.query_params['method']
        section = 'abstract' if 'section' not in self.request.query_params else self.request.query_params['section']
        query = self.request.query_params['query']
        k = 10 if 'k' not in self.request.query_params else int(self.request.query_params['k'])
        k = min(k, 20)
        model = self.get_model(method)
        ls = model.run(query, section, k)
        return Response(ls)

    def get_model(self, method):
        if method == 'fasttext':
            return fasttext_model
        if method == 'boolean':
            return boolean_model
        if method == 'transformers':
            return transformer_model
        if method == 'tfidf':
            return tfidf_model
        if method == 'elastic':
            es_model
        return es_model
