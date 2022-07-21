import spacy
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from spellchecker import SpellChecker


class Preprocess:

    def __init__(self):
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.sno_stemmer = SnowballStemmer('english')
        self.spell = SpellChecker()
        self.nlp = spacy.load('en_core_web_sm')

    def run_spacy(self, raw_text):
        return self.normalize_sentence(self.nlp(raw_text))

    def normalize_sentence(self, tokenized_sent, stopword_removal=True, punctuation_removal=True,
                           lower_case=True, lemmatize=True, minimum_length=2):
        normalized_sent = tokenized_sent

        if stopword_removal:
            normalized_sent = [token for token in normalized_sent
                               if not token.is_stop]
        if punctuation_removal:
            normalized_sent = [token for token in normalized_sent
                               if not token.is_punct]
        if lemmatize:
            normalized_sent = [token.lemma_ for token in normalized_sent]

        if lower_case:
            normalized_sent = [token.lower() for token in normalized_sent]

        if minimum_length > 1:
            normalized_sent = [token for token in normalized_sent
                               if len(token) > minimum_length]

        return ' '.join(normalized_sent)

    def remove_punctuation(self, text):
        return "".join([i for i in text if i not in string.punctuation])

    def tokenize(self, text):
        return word_tokenize(text)

    def remove_stopwords(self, tokens, minimum_length=3):
        return [t for t in tokens if t not in self.stopwords and len(t) >= minimum_length]

    def lower_case(self, tokens):
        return [t.lower() for t in tokens]

    def lemmatize(self, tokens):
        return [self.wordnet_lemmatizer.lemmatize(t) for t in tokens]

    def stem(self, tokens):
        return [self.sno_stemmer.stem(t) for t in tokens]

    def correct_typo(self, tokens):
        return [self.spell.correction(t) if len(self.spell.unknown([t])) > 0 else t for t in tokens]

    def run(self, raw_text, correction=False, stem=False):

        text = self.remove_punctuation(raw_text)

        tokens = self.tokenize(text)

        tokens = self.remove_stopwords(tokens)

        tokens = self.lower_case(tokens)

        if correction:
            tokens = self.correct_typo(tokens)

        if stem:
            tokens = self.stem(tokens)

        tokens = self.lemmatize(tokens)

        return ' '.join(tokens)

    def simple(self, raw_text):
        text = self.remove_punctuation(raw_text)
        tokens = self.tokenize(text)
        tokens = self.lower_case(tokens)
        return ' '.join(tokens)
