import re
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

regex = re.compile(r"\b\w{2,}\b")
stem = SnowballStemmer('english')
stopwords = ENGLISH_STOP_WORDS

def tokenize(text):
    tokens = regex.findall(text)
    tokens = [token for token in tokens if token not in stopwords]
    stems = [stem.stem(token) for token in tokens]
    return stems


def preprocess(text):
    tokens = text.lower().split()
    tokens = [token for token in tokens if token not in stopwords]
    stems = [stem.stem(token) for token in tokens]
    return stems


def make_bigrams(tokens, min_n=1, max_n=2):
    original_tokens = tokens
    if min_n == 1:
        tokens = list(original_tokens)
        min_n += 1

    n_original_tokens = len(original_tokens)
    for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
        for i in range(n_original_tokens - n + 1):
            tokens.append(" ".join(original_tokens[i: i + n]))

    return tokens


