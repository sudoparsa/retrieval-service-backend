import json
import time
from informationretrieval.utils.preprocess import Preprocess


class TFIDFRetrieval:
    def __init__(self,
                 title_tfidf_path='models/TFIDF/title_tfidf.json',
                 df_info_path='models/TFIDF/df_info.json',
                 abstract_tfidf_path='models/TFIDF/abstract_tfidf.json'):
        print('Loading TF-IDF...')
        self.title_tfidf = json.load(open(title_tfidf_path))
        self.abstract_tfidf = json.load(open(abstract_tfidf_path))
        self.df_info = json.load(open(df_info_path))
        self.preprocessor = Preprocess()

    def run_query(self, query, section, k):
        if section == 'title':
            docs = self.title_tfidf
        else:
            docs = self.abstract_tfidf

        query = self.preprocessor.run(query)
        query_terms = query.split()
        N = len(docs)
        results = []
        for i in range(N):
            score = 0
            for term in query_terms:
                tfidf = docs[i].get(term)
                if tfidf is None:
                    continue
                score += tfidf
            results.append((score, i))
        results.sort(key=lambda x: x[0], reverse=True)
        indexs = [x[1] for x in results]
        return indexs[:k]

    def run(self, query, section='title', k=10, query_expansion=False):
        start_time = time.time()
        result = self.run_query(query, section, k)
        print(f'Query: {query}')
        result = self.show(result)
        print(f'Execution time: {time.time() - start_time}')
        return result

    def show(self, indexes):
        result = []
        print()
        print('Similar Papers:')
        for ix, i in zip(indexes, range(len(indexes))):
            print(f'\n{i}.', end='')
            print(f' {self.df_info["title"][ix]}')
            result.append({'title': self.df_info["title"][ix],
                           'url': self.df_info["url"][ix]})
        print()
        return result


tfidf_model = TFIDFRetrieval()
