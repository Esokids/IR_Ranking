import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from ReadFile import open_read_file
from Preprocess import tokenize
from math import log


def make_tf_idf():  # if have any new files
    file, doc = open_read_file()
    df = pd.DataFrame({'text': file}, index=doc)
    v = TfidfVectorizer(smooth_idf=False, sublinear_tf=True, tokenizer=tokenize, ngram_range=(1, 2))
    x = v.fit_transform(df['text'])
    tf_idf = pd.DataFrame(x.toarray(), columns=v.get_feature_names(), index=doc)
    tf_idf.to_csv('tf_idf.csv')


def get_tf_idf():
    df = pd.read_csv('tf_idf.csv', index_col=0)
    return df


def make_tf(lists):
    count_word = dict()
    tf = pd.DataFrame()
    for word in lists:
        count_word[word] = 1 if word not in count_word else count_word[word]+1
    count_word = sorted(count_word.items())
    for word in count_word:
        tf[word[0]] = pd.Series(1 + log(word[1]))
    return tf.drop_duplicates()
