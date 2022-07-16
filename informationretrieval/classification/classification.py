import pickle

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from informationretrieval.health_retrieval import fasttext_model


class BaseClassifier:
    def get_category_title(self, i):
        category_dictionary = {
            'سلامت روان': 0,
            'دهان و دندان': 1,
            'پوست و مو': 2,
            'تغذیه': 3,
            'سلامت خانواده': 4,
            'سلامت جنسی': 5,
            'پیشگیری و بیماریها': 6
        }
        return dict(zip(category_dictionary.values(), category_dictionary.keys()))[i]


class ClassicClassifier(BaseClassifier):
    def predict(self, text):
        emb = fasttext_model.get_text_embeding(text)
        return self.get_category_title(self.model.predict(emb.reshape(1, -1))[0])


class NaiveBayesClassifier(ClassicClassifier):
    def __init__(self):
        self.model = pickle.load(open('models/classification-clustering/naive-bayes.sav', 'rb'))


class LogisticRegressionClassifier(ClassicClassifier):
    def __init__(self):
        self.model = pickle.load(open('models/classification-clustering/logistic-regression.sav', 'rb'))


class TransformerClassifier(BaseClassifier):
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained('models/transformer_model',
                                                                        local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained('models/pretrained-transformer-tokenizer')

    def predict(self, text):
        _input = self.tokenizer(text, truncation=True, padding=True, return_tensors='pt')
        output = self.model(**_input)
        return self.get_category_title(output[0].softmax(1).argmax().item())


transformer_classifier = TransformerClassifier()
logistic_regression_classifier = LogisticRegressionClassifier()
naive_bayes_classifier = NaiveBayesClassifier()
