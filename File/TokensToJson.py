import glob
import json
from nltk.corpus import stopwords
from collections import defaultdict
from nltk import word_tokenize
import nltk.stem as nltk

stop_words = set(stopwords.words('english'))
punctuations = '''!()-[]{};:"'\,<>./?@#$%^&*_~/———`“”‘’'''
stop_words.update(punctuations)


def tokens_to_csv():
    dic = defaultdict(list)
    for filename in glob.glob("*.txt"):
        with open(filename, 'r') as f:
            tokens = word_tokenize(f.read().lower())
            tokens = set([nltk.WordNetLemmatizer().lemmatize(token) for token in tokens])
            tokens = [w for w in tokens if w not in stop_words]
            for word in tokens:
                dic[word].append(filename)

    with open('Tokens.json', 'w') as f:
        json.dump(dic, f)


if __name__ == '__main__':
    tokens_to_csv()
