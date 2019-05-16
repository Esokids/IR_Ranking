import glob
from pathlib import Path
import pandas as pd
import re
import fnmatch
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
stem = SnowballStemmer('english')
stopwords = ENGLISH_STOP_WORDS
regex = re.compile(r"\b\w{2,}\b")


def open_read_file():
    file = list()
    doc = list()
    for filename in glob.glob('.\\File\\*.txt'):
        doc.append(Path(filename).stem)
        with open(filename, 'r') as f:
            file.append(f.read())
    return file, doc


def tokenize(text):
    tokens = regex.findall(text)
    tokens = [token for token in tokens if token not in stopwords]
    stems = [stem.stem(token) for token in tokens]
    return stems


def search_preprocess(keyword, df):
    tokens = df.columns
    keyword = tokenize(keyword)
    search_words = list()

    for i in keyword:
        search_words.extend(fnmatch.filter(tokens, i))

    return search_words


def search(keyword=None, df=None):
    if len(keyword) == 0:
        return False

    temp = pd.DataFrame()
    for i in keyword:
        if i in df:
            temp[i] = pd.Series(df[i].values, index=df.index)
    temp['Sum'] = pd.Series(temp.sum(axis=1), index=df.index)
    temp = temp.sort_values(by=['Sum'], ascending=False)

    idx = temp.index[temp['Sum'] > 0].tolist()
    return idx


def main():
    file, doc = open_read_file()
    df = pd.DataFrame({'text': file}, index=doc)

    v = TfidfVectorizer(smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
    x = v.fit_transform(df['text'])

    tfidf = pd.DataFrame(x.toarray(), columns=v.get_feature_names(), index=doc)
    search_words = search_preprocess('Drones Are Dropping Poison on Rats', tfidf)
    # print(search_words)
    result = search(search_words, tfidf)

    if result is False:
        print("Not found")
    else:
        for i in result:
            print(f"{i}.txt")


if __name__ == '__main__':
    start_time = time()
    main()
    end_time = time()
    print("Total Times : %.4f" % (end_time-start_time))