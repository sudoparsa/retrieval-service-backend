import pickle
import time


class NBClassifier:
    def __init__(self,
                 nb_classifier_path="models/Classification/NB_Classification/nb_classifier.pickle",
                 vectorizer_path="models/Classification/NB_Classification/vectorizer.pk"):
        print('Loading Naive Bayes')
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


nb_classifier = NBClassifier()
