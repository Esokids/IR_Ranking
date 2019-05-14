import glob
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
stem = SnowballStemmer('english')
stopwords = ENGLISH_STOP_WORDS
regex = re.compile(r"\b\w{3,}\b")


def open_read_file():
    file = list()
    for filename in glob.glob('.\\File\\*.txt'):
        with open(filename, 'r') as f:
            file.append(f.read())
    return file


def tokenize(text):
    # tokens = word_tokenize()
    tokens = regex.findall(text)
    tokens = [token for token in tokens if token not in stopwords]
    stems = [stem.stem(token) for token in tokens]
    return stems


def main():
    file = open_read_file()
    df = pd.DataFrame({'text': file})

    v = TfidfVectorizer(smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
    x = v.fit_transform(df['text'])

    tfidf = pd.DataFrame(x.toarray(), columns=v.get_feature_names())
    print(tfidf)


if __name__ == '__main__':
    main()