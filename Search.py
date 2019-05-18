import pandas as pd
import fnmatch
import difflib
from Preprocess import preprocess
from Preprocess import make_bigrams
from TF_IDF import get_tf_idf

df = get_tf_idf()

def search(keyword=None):
    if len(keyword) == 0:
        return False
    keyword = preprocess_search(keyword)
    temp = pd.DataFrame()
    for i in keyword:
        if i in df:
            temp[i] = pd.Series(df[i].values, index=df.index)
    temp['Sum'] = pd.Series(temp.sum(axis=1))
    temp = temp.sort_values(by='Sum', ascending=False)
    idx = temp.index[temp['Sum'] > 0].tolist()
    return idx


def preprocess_search(keyword):
    corpus = df.columns
    keyword = preprocess(keyword)
    keyword = make_bigrams(keyword)
    search_words = list()

    for word in keyword:
        if '*' in word:
            search_words.extend(fnmatch.filter(corpus, word))
        else:
            search_words.extend(difflib.get_close_matches(word, corpus))
    return search_words
