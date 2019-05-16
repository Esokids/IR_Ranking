import glob
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer
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


def search(keyword=None, data=None):
    if keyword is None:
        return False

    temp = pd.DataFrame()

    for i in keyword:
        if i in data:
            temp[i] = pd.Series(data[i].values, index=data.index)
    temp['Sum'] = pd.Series(temp.sum(axis=1), index=data.index)
    temp = temp.sort_values(by=['Sum'], ascending=False)

    idx = temp.index[temp['Sum'] > 0].tolist()
    return idx


def main():
    file = open_read_file()
    df = pd.DataFrame({'text': file})

    v = TfidfVectorizer(smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
    x = v.fit_transform(df['text'])

    tfidf = pd.DataFrame(x.toarray(), columns=v.get_feature_names())
    #print(tfidf)

    result = search(tokenize('150 year'), tfidf)
    for i in result:
        print(f"{int(i)+1}.txt")


if __name__ == '__main__':
    main()