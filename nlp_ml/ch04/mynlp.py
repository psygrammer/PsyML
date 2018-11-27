import nltk
import string

class TextCorpusReader(object):

    def __init__(self, path):
        lst = os.listdir(path)
        self._fnlist = [os.path.join(path, fn) for fn in lst 
                                                    if 'txt' in fn]
    

    def docs(self):
        corpus = []
        for fn in self._fnlist : 
            with open(fn) as f :
                for line in f :
                    corpus.append(line.strip())
        return corpus

# Tokenization function
def tokenize(text):
    stem = nltk.stem.SnowballStemmer('english')
    text = text.lower()

    for token in nltk.word_tokenize(text):
        if token in string.punctuation: continue
        yield stem.stem(token)
   
        
def nltk_frequency_vectorize(corpus):

    # The NLTK frequency vectorize method
    from collections import defaultdict

    def vectorize(doc):
        features = defaultdict(int)

        for token in tokenize(doc):
            features[token] += 1

        return features

    return map(vectorize, corpus)


def sklearn_frequency_vectorize(corpus):
    # The Scikit-Learn frequency vectorize method
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(corpus)


def gensim_frequency_vectorize(corpus):
    # The Gensim frequency vectorize method
    import gensim

    corpus_g = [list(tokenize(doc)) for doc in corpus]
    id2word = gensim.corpora.Dictionary(corpus_g)

    return [
        id2word.doc2bow(doc) for doc in corpus_g
    ]
