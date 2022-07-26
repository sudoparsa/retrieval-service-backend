import json
import time
from informationretrieval.utils.preprocess import Preprocess
from informationretrieval.utils.expansion import Rocchio
import numpy as np
import os
from gensim.models import KeyedVectors
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity


class FastTextRetrieval:
    def __init__(self,
                 word_vectors_path='models/FastText/word_vectors.kv',
                 ngram_path='models/FastText/word_vectors.kv.vectors_ngrams.npy',
                 embedding_path='models/FastText/doc_embedding.json',
                 term2idf_path='models/FastText/term2idf.json'):
        print('Loading FastText...')
        if not os.path.isfile(ngram_path):
            self.download_model(ngram_path)
        self.word_vectors = KeyedVectors.load(word_vectors_path)
        self.doc_embedding = json.load(open(embedding_path))
        self.term2idf = json.load(open(term2idf_path))
        self.preprocessor = Preprocess()
        self.embedding_size = len(self.doc_embedding['embedding'][0])

    def download_model(self, path):
        id = '1--kLTUB5UBF_ED1UhGNrkagduXfUoNtA'
        os.chdir(os.path.dirname(path))
        os.system('gdown 1--kLTUB5UBF_ED1UhGNrkagduXfUoNtA')
        os.chdir('../' * len(os.path.dirname(path).split('/')))

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
        vectors = []
        weights = []
        for term in text.split():
            if term in self.term2idf.keys():
                vectors.append(np.array(self.word_vectors[term]))
                weights.append(self.term2idf[term])
        if len(weights) == 0:
            return np.zeros(self.embedding_size)
        weights = np.array(softmax(weights))
        vectors = np.array(vectors)
        return weights @ vectors

    def most_similar(self, query, is_query_embedded, k):
        if not is_query_embedded:
            clean_query = self.preprocessor.run(query)
            query_embedding = self.embed(clean_query).reshape(1, -1)
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
            query = Rocchio(self, query)
        indx, scores = self.most_similar(query, query_expansion, k)
        result = self.show(indx, scores=scores)
        print(f'Execution time: {time.time() - start_time}')
        return result


fasttext_model = FastTextRetrieval()
