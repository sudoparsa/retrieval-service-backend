import json
import time
from informationretrieval.utils.preprocess import Preprocess
from informationretrieval.utils.expansion import Rocchio
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class TransformerRetrieval:
    def __init__(self,
                 model_path='models/SentenceTransformers/model/',
                 embedding_path='models/SentenceTransformers/doc_embedding.json'):
        print('Loading SentenceTransformer...')
        self.model = SentenceTransformer(model_path)
        self.doc_embedding = json.load(open(embedding_path))
        self.preprocessor = Preprocess()

    def show(self, indexes, scores):
        result = []
        print()
        print('Similar Papers using Cosine Similarity:')
        for ix, i in zip(indexes, range(len(indexes))):
            print(f'\n{i}.', end='')
            print(f' {self.doc_embedding["title"][ix]}')
            print(f'Cosine Similarity : {scores[ix]}')
            result.append({'title': self.doc_embedding["title"][ix],
                           'url': self.doc_embedding["url"][ix],
                           'score': scores[ix]})
        print()
        return result

    def embed(self, text):
        query = self.preprocessor.run(text)
        query_embedding = self.model.encode(query)
        return query_embedding

    def most_similar(self, query, is_query_embedded, k):
        if not is_query_embedded:
            query_embedding = self.embed(query).reshape(1, -1)
        else:
            query_embedding = query
        embeddings = self.doc_embedding['embedding']
        cosine_scores = cosine_similarity(query_embedding, embeddings)[0]
        similar_ix = np.argsort(cosine_scores)[::-1][:k]
        return similar_ix, cosine_scores

    def run(self, query, section='abstract', k=10, query_expansion=False):
        start_time = time.time()
        print(f'Query: {query}')
        if query_expansion:
            query = Rocchio(self, query, k)
        indx, scores = self.most_similar(query, query_expansion, k)
        result = self.show(indx, scores=scores)
        print(f'Execution time: {time.time() - start_time}')
        return result


transformer_model = TransformerRetrieval()
