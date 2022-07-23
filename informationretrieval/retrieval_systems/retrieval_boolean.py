import json
import time
from informationretrieval.utils.preprocess import Preprocess


class BooleanRetrieval:
    def __init__(self,
                 inverted_indices_path='models/Boolean/inverted_indices.json',
                 df_info_path='models/Boolean/df_info.json'):
        print('Loading Boolean...')
        self.inverted_indices = json.load(open(inverted_indices_path))
        self.df_info = json.load(open(df_info_path))
        self.preprocessor = Preprocess()

    def bool_query(self, query, section, k):
        if section == 'title':
            query = self.preprocessor.run(query)
        elif section == 'author':
            query = self.preprocessor.simple(query)
        doc_list = []
        for word in query.split():
            if word in self.inverted_indices[section].keys():
                doc_list += self.inverted_indices[section][word]
        doc_list = sorted(doc_list, key=doc_list.count, reverse=True)
        doc_list = list(dict.fromkeys(doc_list))
        return doc_list[:k]

    def run(self, query, section='title', k=10, query_expansion=False):
        start_time = time.time()
        if section != 'title':
            section = 'author'
        result = self.bool_query(query, section, k)
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


boolean_model = BooleanRetrieval()
