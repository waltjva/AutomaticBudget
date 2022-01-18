# import model_prep
from nltk.tokenize.toktok import ToktokTokenizer
import re
# from contractions import CONTRACTION_MAP
import unicodedata
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

mlb = MultiLabelBinarizer()

nltk.download('wordnet')

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import OneHotEncoder
# from tensorflow.keras.preprocessing.sequence import pad_sequences

lemmatizer = WordNetLemmatizer()

nltk.download('wordnet')
nltk.download('omw-1.4')
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

class content_encoder:
    def __init__(self, content):
        self.content = content

    def remove_stopwords(self, text):
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        filtered_tokens = [token for token in tokens if token not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    def data_corpus_create(self, content):
        data_corpus = set()
        for row in content:
            for word in row.split(" "):
                if word not in data_corpus:
                    data_corpus.add(word)
        return sorted(data_corpus)

    def onehot_encode_words(self):
        content = self.content
        content = content.apply(lambda x: re.sub(r"[^a-zA-Z0-9]+", " ", x))
        content = content.apply(lambda x: x.lower())
        content = content.apply(lambda x: x.strip())
        # create corpus with frequency counts to see what can stay and what can depart
        content = pd.Series(sorted(content))
        # Create a corpus for all the words
        content = content.apply(self.remove_stopwords)
        content = content.apply(lambda x: lemmatizer.lemmatize(x))
        content = content.apply(lambda x: re.sub("\d+", "", x))
        content = content.apply(lambda x: re.sub("  ", " ", x))
        content = content.apply(lambda x: x.strip())
        content = content.apply(lambda x: re.sub("  ", " ", x))

        # Create dictionary for each unique word, += 1 for each new encounter
        d = {}
        for row in content:
            for word in row.split(" "):
                if word not in d.keys():
                    d[word] = 1
                else:
                    d[word] += 1

        word_freq_counts = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}

        ref = list(d.values())

        def remove_dict_dat(sentence):
            s = sentence.split(" ")
            for word in s:
                if d[word] < 5:
                    s.remove(word)
                elif d[word] > 19:
                    s.remove(word)
            return " ".join(s)

        content = content.apply(remove_dict_dat)
        content = content.apply(lambda x: x.strip())
        content = content.apply(lambda x: lemmatizer.lemmatize(x))

        # converting text to integers
        token_docs = [list(set(doc.split(" "))) for doc in content]
        sentences = pd.Series(token_docs)

        return pd.DataFrame(mlb.fit_transform(sentences), columns=mlb.classes_)