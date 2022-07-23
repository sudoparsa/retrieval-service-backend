import json
import time
from informationretrieval.utils.preprocess import Preprocess
import numpy as np
from sentence_transformers import SentenceTransformer, util
import pickle


class KmeansClustering:
    def __init__(self,
                 model_path='models/SentenceTransformers/model/',
                 embedding_path='models/Clustering/doc_embedding.json',
                 kmeans_path='models/Clustering/kmeans.pkl',
                 result_path='models/Clustering/result.json'):
        print('Loading KMeans...')
        self.model = SentenceTransformer(model_path)
        self.doc_embedding = json.load(open(embedding_path))
        self.kmeans = pickle.load(open(kmeans_path, "rb"))
        self.clustering_metrics = json.load(open(result_path))
        self.preprocessor = Preprocess()

    def show(self, indexes, scores, cluster_embeddings):
        result = []
        print()
        print('Similar Papers using Cosine Similarity:')
        for ix, i in zip(indexes, range(len(indexes))):
            print(f'\n{i}.', end='')
            print(f' {cluster_embeddings["title"][ix]}')
            print(f'Label : {cluster_embeddings["label"][ix]}')
            print(f'Cosine Similarity : {scores[ix]}')
            result.append({'title': cluster_embeddings["title"][ix],
                           'url': cluster_embeddings["url"][ix],
                           'label': cluster_embeddings["label"][ix],
                           'score': scores[ix]})
        print()
        return result

    def get_cluster_embeddings(self, cluster):
        ix = []
        for i, emb in enumerate(self.doc_embedding['embedding']):
            if cluster == self.kmeans.predict(np.array(list(map(np.float32, emb))).reshape(1, -1)):
                ix.append(i)
        r = {}
        for key in self.doc_embedding.keys():
            r[key] = [self.doc_embedding[key][i] for i in ix]
        return r

    def most_similar(self, query, k):
        query = self.preprocessor.run_spacy(query)
        query_embedding = self.model.encode(query)
        query_cluster = self.kmeans.predict(query_embedding.reshape(1, -1))
        cluster_embeddings = self.get_cluster_embeddings(query_cluster)
        embeddings = cluster_embeddings['embedding']
        cosine_scores = util.dot_score(query_embedding, embeddings).detach().cpu().numpy()[0]
        similar_ix = np.argsort(cosine_scores)[::-1][:k]
        return similar_ix, cosine_scores, cluster_embeddings

    def run(self, query, k=10):
        start_time = time.time()
        print(f'Query: {query}')
        indx, scores, cluster_embeddings = self.most_similar(query, k)
        result = self.show(indx, scores, cluster_embeddings)
        print(f'Execution time: {time.time() - start_time}')
        return result


kmeans_clustering_model = KmeansClustering()
