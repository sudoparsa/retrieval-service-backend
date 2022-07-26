import pickle
import time
import os
import numpy as np
import tensorflow
from transformers import TextClassificationPipeline
from transformers import TFAutoModelForSequenceClassification


class NBClassifier:
    def __init__(self,
                 nb_classifier_path="models/Classification/NB_Classification/nb_classifier.pickle",
                 vectorizer_path="models/Classification/NB_Classification/vectorizer.pk"):
        print('Loading Naive Bayes...')
        self.nb_classifier = pickle.load(open(nb_classifier_path, 'rb'))
        self.vectorizer = pickle.load(open(vectorizer_path, 'rb'))
        self.label2field = {
            0: "Computer Science",
            1: "Engineering",
            2: "Medicine",
            3: "Psychology",
            4: "Materials Science",
        }

    def nb_classify(self, query):
        query_dtm = self.vectorizer.transform([query])
        result_class = self.nb_classifier.predict(query_dtm)[0]
        result_prob = self.nb_classifier.predict_proba(query_dtm)[0][result_class]
        return result_class, result_prob

    def run(self, query):
        start_time = time.time()
        result_class, result_prob = self.nb_classify(query)
        print(f"Query: {query}")
        print(f"Predicted Category: {[self.label2field[result_class]]} With Probability: {result_prob}")
        print(f"Execution time: {time.time() - start_time}")
        return {'label': self.label2field[result_class],
                'score': result_prob}


class TransformerClassifier:
    def __init__(self,
                 transformer_classifier_path="models/Classification/Transformer_Classification/",
                 vectorizer_path="models/Classification/Transformer_Classification/tokenizer.pk"):
        print('Loading Transformer Classifier...')
        if not os.path.isfile(transformer_classifier_path + "tf_model.h5"):
            self.download_model(transformer_classifier_path)
        self.transformer_model = TFAutoModelForSequenceClassification.from_pretrained(transformer_classifier_path)
        self.tokenizer = pickle.load(open(vectorizer_path, "rb"))
        self.pipe = TextClassificationPipeline(model=self.transformer_model,
                                               tokenizer=self.tokenizer,
                                               return_all_scores=True)
        self.label2field = {
            0: "Computer Science",
            1: "Engineering",
            2: "Medicine",
            3: "Psychology",
            4: "Materials Science",
        }

    def download_model(self, path):
        id_model = "17jWbiMqIJ1ed_3yNeonsYu2Q3fUb77qF"
        id_config = "1rbSJVdZ9SudXlG65_fx9WsAucLsPWxmw"
        os.chdir(path)
        os.system('gdown 17jWbiMqIJ1ed_3yNeonsYu2Q3fUb77qF')
        os.system('gdown 1rbSJVdZ9SudXlG65_fx9WsAucLsPWxmw')
        os.chdir("../" * len(os.path.dirname(path).split("/")))

    def transformer_classify(self, query):
        result = self.pipe(query)
        result_class = np.argmax([x['score'] for x in result])
        result_prob = result[result_class]['score']
        return result_class, result_prob

    def run(self, query):
        start_time = time.time()
        result_class, result_prob = self.transformer_classify(query)
        print(f"Query: {query}")
        print(f"Predicted Category: {[self.label2field[result_class]]} With Probability: {result_prob}")
        print(f"Execution time: {time.time() - start_time}")
        return {'label': self.label2field[result_class],
                'score': result_prob}


nb_classifier = NBClassifier()
transformer_classifier = TransformerClassifier()
