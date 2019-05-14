import glob
from nltk.corpus import stopwords
from collections import defaultdict
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
punctuations = '''!()-[]{};:"'\,<>./?@#$%^&*_~/———`“”‘’'''
stop_words.update(punctuations)


def tokens_to_csv():
    dic = defaultdict(list)
    for filename in glob.glob("*.txt"):
        with open(filename, 'r') as f:
            tokens = word_tokenize(f.read().lower())
            tokens = set([WordNetLemmatizer().lemmatize(token) for token in tokens])
            tokens = [w for w in tokens if w not in stop_words]
            for word in tokens:
                dic[word].append(filename)

    with open('Tokens.csv', 'w') as f:
        header = "Token,PostingList\n"
        f.write(header)
        for key, value in dic.items():
            data = ""
            for e in value:
                data += str(e) + " "
            f.write(f"{key},{data}\n")


if __name__ == '__main__':
    tokens_to_csv()
