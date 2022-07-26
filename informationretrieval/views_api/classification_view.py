from rest_framework.response import Response
from rest_framework.views import APIView
from informationretrieval.classification import nb_classifier, transformer_classifier


class ClassificationView(APIView):

    def get(self, request, *args, **kwargs):
        method = self.request.query_params['method']
        model = self.get_model(method)
        query = self.request.query_params['query']
        return Response(model.run(query))

    def get_model(self, method):
        if method == 'naive_bayes':
            return nb_classifier
        if method == 'transformers':
            return transformer_classifier
        return transformer_classifier
