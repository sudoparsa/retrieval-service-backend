import json
import pickle
import pandas as pd

import numpy as np

from informationretrieval.health_retrieval import fasttext_model


class KmeansClustering:
    def __init__(self):
        self.kmeans = pickle.load(open("models/classification-clustering/kmeans.pkl", "rb"))
        with open('models/classification-clustering/clustering_embedding.json', 'r', encoding="utf-8") as f:
            self.all_data = json.loads(f.read())
        self.df = pd.DataFrame(self.all_data)

    def predict(self, text, k=10):
        emb = fasttext_model.get_text_embeding(text)
        predicted_cluster = self.kmeans.predict([emb / np.linalg.norm(emb)])[0]
        df_same_cluster = self.df[self.df['cluster'] == predicted_cluster].reset_index(drop=True)
        ten_most_similar_index = list(
            np.argsort(df_same_cluster.apply(lambda x: self.cosine_similarity(x['embedding'], emb), axis=1))[-k:][
            ::-1])
        result = []
        for i in ten_most_similar_index:
            result.append({'title': df_same_cluster['title'][i], 'url': df_same_cluster['link'][i]})
        return result

    def cosine_similarity(self, vector_1: np.ndarray, vector_2: np.ndarray) -> float:
        return np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))


kemans_clustering_model = KmeansClustering()
