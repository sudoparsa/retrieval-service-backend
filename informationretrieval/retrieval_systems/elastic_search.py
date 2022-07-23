import time
import pandas as pd
from elasticsearch import Elasticsearch
from subprocess import Popen


class ElasticSearch:
    def __init__(self,
                 data_path='data/semanticscholar.json',
                 elasticsearch_path='..\\elasticsearch-7.3.2\\bin\\elasticsearch.bat'):
        print("Starting Elasticsearch Server...")
        self.hosts = ['http://localhost:9200']
        self.index = 'articles'
        self.__start_elasticsearch_server(elasticsearch_path)
        print("Creating Index and Adding Data...")
        self.create_index(self.index, data_path)

    def __start_elasticsearch_server(self, elasticsearch_path):
        Popen(elasticsearch_path)
        time.sleep(35)

    def create_index(self, index, data_path):
        df = pd.read_json(data_path)
        es_data = self.prepare_es_data(index=index, df=df)
        self.index_es_data(index=index, es_data=es_data)

    def prepare_es_data(self, index, df):
        records = df.to_dict(orient="records")
        es_data = []
        for idx, record in enumerate(records):
            meta_dict = {
                "index": {
                    "_index": index,
                    "_id": idx
                }
            }
            es_data.append(meta_dict)
            es_data.append(record)
        return es_data

    def index_es_data(self, index, es_data):
        es_client = Elasticsearch(hosts=self.hosts)
        if es_client.indices.exists(index=index):
            print("deleting the '{}' index.".format(index))
            res = es_client.indices.delete(index=index)
            print("Response from server: {}".format(res))

        print("creating the '{}' index.".format(index))
        res = es_client.indices.create(index=index)
        print("Response from server: {}".format(res))

        print("bulk index the data")
        res = es_client.bulk(index=index, body=es_data, refresh=True)
        print("Errors: {}, Num of records indexed: {}".format(res["errors"], len(res["items"])))

    def show(self, results):
        res = []
        print()
        print('Results:')
        for ix, paper in enumerate(results):
            print(f'\n{ix}.', end='')
            print(f" {paper['_source']['title']}")
            print(f"Score: {paper['_score']}")
            res.append({'title': paper['_source']['title'],
                        'url': paper['_source']['url'],
                        'score': paper['_score']})
        print()
        return res

    def run(self, query, section='title', k=10, query_expansion=False):
        start_time = time.time()
        print(f'Query: {query}')
        es_client = Elasticsearch(hosts=self.hosts)
        response = es_client.search(index=self.index, query={"match": {section: query}})['hits']['hits'][:k]
        result = self.show(response)
        print(f'Execution time: {time.time() - start_time}')
        return result


es_model = ElasticSearch()
