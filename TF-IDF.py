import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')



class tf_idf_embedding:

    def __init__(self, lyrics_csv_path = "lyrics-data-toy.csv"):

        self.lyrics_df = pd.read_csv(lyrics_csv_path)[["Lyric"]]
        self.lemmatize_stemming_lyrics = pd.DataFrame(columns = ["lyric"])

        self.embedding_matrix = pd.DataFrame()
        self.tf_idf_vectorizer = None
        self.tf_idf = None

        self.create_lemmatize_stemming_lyrics_df()
        self.create_tf_idf_vectorizer(max_df=0.01, min_df= 1)

    def print_current_time(self, text = ""):

        import datetime
        import pytz
        current_time = datetime.datetime.now(pytz.timezone('Israel'))
        if text!= "":
            print(text)
        print("The current time is: ")
        print(current_time)
        print()

    def lemmatize_stemming(self,token):
        stemmer = PorterStemmer()
        return stemmer.stem(WordNetLemmatizer().lemmatize(token, pos='v'))

    def preprocess_lyric(self, lyric):
        prep_lyric= ""
        for token in gensim.utils.simple_preprocess(lyric):
                if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:
                    prep_lyric += (self.lemmatize_stemming(token)) + " "

        return  prep_lyric

    def create_lemmatize_stemming_lyrics_df(self):

        self.print_current_time("creating lemmatize stemming lyrics df")

        for i in range(len(self.lyrics_df)) :
             lyric =   self.lyrics_df.loc[i][0]
             prep_lyric = self.preprocess_lyric(lyric)[:-1]
             self.lemmatize_stemming_lyrics.loc[i]= prep_lyric

        self.print_current_time("finished to create lemmatize stemming lyrics df")

    def create_tf_idf_vectorizer(self, max_df, min_df= 10):

        from sklearn.feature_extraction.text import TfidfVectorizer

        self.print_current_time("creating TF-IDF vectorizer. max_df= " + str(max_df) + " min_df= " + str(min_df))

        self.tf_idf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words= 'english' )
        self.tf_idf =  self.tf_idf_vectorizer.fit_transform(self.lemmatize_stemming_lyrics['lyric'].values.astype('U'))

        self.print_current_time(" finish creating TF-IDF vectorizer")

        self.embedding_matrix = pd.DataFrame(data =  self.tf_idf.todense(), columns = self.tf_idf_vectorizer.get_feature_names()  )

        print(self.tf_idf_vectorizer.get_feature_names())



sasi =    tf_idf_embedding()
