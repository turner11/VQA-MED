from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
#nltk.download('all')
nltk.download('punkt')
# nltk.download()
import sys
# ''' This is for avoidong error of: 'ascii' codec can't decode byte 0xc2'''
# reload(sys)
# sys.setdefaultencoding('utf8')


class Preprocessor():
    _lemmatizer = WordNetLemmatizer()
    _stoplist = stopwords.words('english')


    def __init__(self):
        pass

    @classmethod
    def init_lists(cls,dataPath, label):
        with open(dataPath) as file:
            lines = [line.strip() for line in file]
        # we need at least 1 tab to differentiate content and label
        lines = [line for line in lines if line.count('\t') > 0]

        corpus = [line.split('\t', 1)[1] for line in lines if line.startswith(label)]
        return corpus


    @classmethod
    def tokeniz(cls,str):
        # tokenizers - spllitting the text by white spaces and punctuation marks
        # lemmatizers - linking the different forms of the same word (for example, price and prices, is and are)
        res = [WordNetLemmatizer.lemmatize(cls._lemmatizer,word.lower()) for word in word_tokenize(str)
                 if 2 < len(word) < 50
                ]
        return res
        #return [WordNetLemmatizer.lemmatize("Hello World".lower()) for word in word_tokenize(str)]

    @classmethod
    def batch_tokeniz(cls,corpus):
        return [Preprocessor.tokeniz(line) for line in corpus]