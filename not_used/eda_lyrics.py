import numpy as np
import pandas as pd
import time

# lyrics_file = r"C:\Anat\University\NLP\Project\lyrics-data.csv"
# artists_file = r"C:\Anat\University\NLP\Project\artists-data.csv"
#
# # read files
# df_lyrics = pd.read_csv(lyrics_file)
# df_artists = pd.read_csv(artists_file).sort_values('Artist')
# df_artists = df_artists.fillna(' ')
# # df_artists.set_index('Link', inplace=True)
# # df_lyrics.set_index('ALink', inplace=True)
#
# # filter english lyrics
# df_lyrics_english = df_lyrics[df_lyrics.Idiom == 'ENGLISH'].copy()
# df_lyrics_english.drop_duplicates(inplace=True)
# df_lyrics_english.to_csv(r'C:\Anat\University\NLP\Project\english-lyrics-data.csv')
#
# # filter rap artists
# # df_artists['is_rap'] = False
# # for index, row in df_artists.iterrows():
# #     genres = row['Genres'].split(";")
# #     is_rap = "Rap" in genres or " Rap" in genres
# #     df_artists.loc[index, 'is_rap'] = is_rap
#
# # save new files of rappers
# # df_artists.drop_duplicates(subset=['Artist'], inplace=True)
# # rap_artists = pd.read_csv(r'C:\Anat\University\NLP\Project\rap-artists-data.csv')
# # rap_lyrics = df_lyrics.loc[rap_artists.Link].reset_index()
# # rap_lyrics.to_csv(r'C:\Anat\University\NLP\Project\rap-lyrics-data.csv')
#
# # remove feature songs and remix
# # rap_lyrics['is_feat'] = False
# df_lyrics_english['is_remix'] = False
# df_lyrics_english['is_cover'] = False
#
# for index, row in df_lyrics_english.iterrows():
#     sname = row['SName']
#     lyrics = row['Lyric']
#     slink = row['SLink']
#     # is_feat = ("feat" in sname or "Feat" in sname or "ft" in sname or "Ft" in sname or "(with" in sname or
#     #            "(With" in sname or "feat" in lyrics or "Feat" in lyrics or "ft" in lyrics or "Ft" in lyrics or
#     #            "feat" in slink or "Feat" in slink)
#     is_remix = "remix" in sname or "Remix" in sname or "remix" in slink or "Remix" in slink
#     is_cover = "cover" in sname or "Cover" in sname or "cover" in slink or "Cover" in slink
#     # rap_lyrics.loc[index, 'is_feat'] = is_feat
#     df_lyrics_english.loc[index, 'is_remix'] = is_remix
#     df_lyrics_english.loc[index, 'is_cover'] = is_cover
#
# english_lyrics_no_dup = df_lyrics_english[~df_lyrics_english.is_remix & ~df_lyrics_english.is_cover].copy()
# english_lyrics_no_dup.to_csv(r'C:\Anat\University\NLP\Project\english_lyrics_data_no_dup.csv')
# artists = ['Elvis Presley', 'Bee Gees', 'Bob Dylan', 'Van Morrison', 'Neil Young', 'Bruce Springsteen',
#            'Elvis Costello', 'David Bowie', 'Rod Stewart', 'Rolling Stones', 'Eric Clapton', 'Britney Spears',
#            'Celine Dion', 'Bon Jovi', 'The Beach Boys', 'The Beatles', 'Madonna', 'U2', 'Beck', 'Alice Cooper']


##############################################################
import lyricsgenius as lg
import numpy as np
import pandas as pd
import time

# genius = lg.Genius('EqvfihkKlnV3aEGQ9rrMuPvDFLoqGOUInOCk6skS5iz0xeJTupbzOXkljeylFQ9G',
#                    skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"], remove_section_headers=True)

# for name in artists[1:]:
#     artist = genius.search_artist(name, get_full_info=False)
#     artist.save_lyrics(extension='json')
#     print(f"Songs grabbed:{artist.num_songs}")

    # except Exception as e:
    #     print(f"some exception at {name}")
    #     print(e)
    #     # artist.save_lyrics(extension='txt')
    #     # artist.save_lyrics(extension='json')
    #     time.sleep(5 * 60)


##########################################################
import glob
import json
import pandas as pd
from pandas import DataFrame as df

# json_files = glob.glob(r"C:\Anat\University\NLP\Project\lyrics\*.json")
# all_songs = df()
#
# for json_file in json_files:
#     with open(json_file, encoding="utf-8") as j_file:
#         artist = json.load(j_file)
#     songs = df(artist.get('songs'))
#     all_songs = all_songs.append(songs, ignore_index=True)
#
# all_songs.to_csv(r"C:\Anat\University\NLP\Project\all_songs.csv")
# is_feature = all_songs['title'] != all_songs['title_with_featured']
# all_songs_no_dup = all_songs.drop_duplicates(subset=['lyrics'], keep='first')
# all_songs_no_dup.to_excel(r"C:\Anat\University\NLP\Project\all_songs_no_dup.xlsx")
# all_songs_no_dup.sort_values(by=['artist', 'title'], inplace=True)

def diff_letters(a,b):
    return sum ( a[i] == b[i] for i in range(min(len(a), len(b))) )

if __name__ == '__main__':
    all_songs_no_dup = pd.read_csv(r"/Project/all_songs_no_dup.csv")
    # all_songs_no_dup = all_songs_no_dup.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1', 'Unnamed: 0.1.1.1.1'], axis=1)
    all_songs_no_dup.sort_values(by=['artist', 'title'], inplace=True)

    for i in range(len(all_songs_no_dup)-1):
        index = all_songs_no_dup.index[i+1]
        row1 = all_songs_no_dup.iloc[i]
        row2 = all_songs_no_dup.iloc[i + 1]
        first_lyrics = row1['title']
        second_lyrics = row2['title']
        errors = diff_letters(first_lyrics, second_lyrics)
        error_percent = errors / min(len(first_lyrics), len(second_lyrics))
        all_songs_no_dup.loc[index, 'diff_percent'] = error_percent

    a=1
    duplicate_songs = (all_songs_no_dup.diff_percent == 1).values
    all_songs_no_dup = all_songs_no_dup[~duplicate_songs]
all_songs_no_dup.to_excel(r"C:\Anat\University\NLP\Project\all_songs_no_dup.xlsx")
artist_count = all_songs_no_dup.groupby('artist').count()
artist_count['avg_words'] = all_songs_no_dup.groupby('artist').max()['words_len']


all_songs_no_dup_new = all_songs_no_dup.drop(['Unnamed: 0', 'words', 'words_len'], axis=1)
all_songs_no_dup_new = all_songs_no_dup_new[~(all_songs_no_dup_new.artist == 'Céline Dion')]
all_songs_no_dup_new.to_excel(r"C:\Anat\University\NLP\Project\all_songs_no_dup_new.xlsx")

df_songs = pd.read_excel(r"C:\Anat\University\NLP\Project\all_songs_no_dup.xlsx")
artist_count['max_words'] = (df_songs['words_len'] > 512).sum()

import os
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))  # This is your Project Root
data_file = os.path.join(r'C:\Anat\University\NLP\nlp_final_project', r'data\all_songs_nineteen_artists.csv')
df_songs = pd.read_csv(data_file)
df_songs_five_artists = df_songs[df_songs.Artist.isin(['Alice Cooper', 'Beck', 'Bee Gees', 'Bob Dylan', 'Bon Jovi']).values].copy()
df_songs_five_artists.to_csv(r'C:\Anat\University\NLP\nlp_final_project\data\songs_five_artists.csv')
