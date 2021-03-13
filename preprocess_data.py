import pandas as pd
from sklearn.preprocessing import LabelEncoder



class preprocess_data:

    def __init__(self,song_csv_path, max_word):
        self.df_songs = pd.read_csv(song_csv_path)
        self.filtered_df_songs = pd.DataFrame()
        self.max_words = max_word

        self.add_labels()
        self.delete_songs_more_n_words(self.max_words)




    def add_labels(self):

        labelencoder = LabelEncoder()
        self.df_songs["label"]=labelencoder.fit_transform(self.df_songs["Artist"])

    def delete_songs_more_n_words(self, max_n_words):

        self.df_songs['words_num'] =   self.df_songs.Lyrics.apply(lambda x: len(str(x).replace("\n"," " ).split(' ')))
        # from collections import Counter
        # self.df_songs['words_num'] =self.df_songs['lyrics'].apply(lambda x: Counter(" ".join(x).split(" ")).items())
        self.filtered_df_songs =  self.df_songs[self.df_songs['words_num'] <=max_n_words].reset_index()













