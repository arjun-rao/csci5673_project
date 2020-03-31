from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from IPython import embed

class FeatureExtractor:
    def __init__(self):
        # Vectorization parameters

        # Range (inclusive) of n-gram sizes for tokenizing text.
        self.NGRAM_RANGE = (1, 2)

        # Limit on the number of features. We use the top 1k features.
        self.TOP_K = 500

        # Whether text should be split into word or character n-grams.
        # One of 'word', 'char'.
        self.TOKEN_MODE = 'word'

        # Minimum document/corpus frequency below which a token will be discarded.
        self.MIN_DOCUMENT_FREQUENCY = 2

        # Limit on the length of text sequences. Sequences longer than this
        # will be truncated.
        self.MAX_SEQUENCE_LENGTH = 500

        self.vectorizer = None
        self.selector = None

    def fit(self, train_texts, train_labels):
        # Create keyword arguments to pass to the 'tf-idf' vectorizer.
        self.kwargs = {
                'ngram_range': self.NGRAM_RANGE,  # Use 1-grams + 2-grams.
                'dtype': 'int32',
                'strip_accents': 'unicode',
                'decode_error': 'replace',
                'analyzer': self.TOKEN_MODE,  # Split text into word tokens.
                'min_df': self.MIN_DOCUMENT_FREQUENCY,
        }

        self.vectorizer = TfidfVectorizer(**self.kwargs)
        # Learn vocabulary from training texts and vectorize training texts.
        x_train = self.vectorizer.fit_transform(train_texts)
        # Select top 'k' of the vectorized features.
        self.selector = SelectKBest(f_classif, k=self.TOP_K)
        self.selector.fit(x_train, train_labels)
        x_train = self.selector.transform(x_train)
        x_train = x_train.astype('float32')
        return x_train


    def transform(self, val_texts):
        # Vectorize validation texts.
        x_val = self.vectorizer.transform(val_texts)
        x_val = self.selector.transform(x_val)
        x_val = x_val.astype('float32')
        return x_val
