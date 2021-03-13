from transformers import BertTokenizer
import pandas as pd
import torch
import re



class tokenizing:

    def __init__(self, texts_csv_path ):

        self.df_songs = pd.read_csv(texts_csv_path, header= 0)
        self.max_embed_batch_len =  512
        self.songs_dict = {}

        self.create_tokenizer_instance()



        #self.tokenizer = None


    def create_tokenizer_instance(self):

        # Load the BERT tokenizer.
        print('Loading BERT tokenizer...')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def  tokenizing_batch (self,batch, add_special_tokens=True):


         # For every batch/song

             # `encode_plus` will:
             #   (1) Tokenize the batch.
             #   (2) Prepend the `[CLS]` token to the start.
             #   (3) Append the `[SEP]` token to the end.
             #   (4) Map tokens to their IDs.
             #   (5) Pad or truncate the sentence to `max_length`
             #   (6) Create attention masks for [PAD] tokens.
         encoded_dict = self.tokenizer.encode_plus(
                 batch,  # Sentence to encode.
                 add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                 max_length= self.max_embed_batch_len,  # Pad & truncate all sentences.
                 pad_to_max_length=True,
                 return_attention_mask=True,  # Construct attn. masks.
                 return_tensors='pt'  # Return pytorch tensors.
             )

         return encoded_dict


    # def choose_name(self, encoded_dict):
    #     # Add the encoded sentence to the list.
    #  input_ids = (encoded_dict['input_ids'])
    #
    #     # And its attention mask (simply differentiates padding from non-padding).
    #  attention_masks=(encoded_dict['attention_mask'])
    #
    #     # Convert the lists into tensors.
    #
    # input_ids = torch.cat(input_ids, dim=0)
    # attention_masks = torch.cat(attention_masks, dim=0)
    # labels = torch.tensor(labels)
    #
    # # Print sentence 0, now as a list of IDs.
    # print('Original: ', sentences[0])
    # print('Token IDs:', input_ids[0])




    def tokenize_each_song(self):




        for i in range(len(self.df_songs)):


           print(i)
           key = ( self.df_songs.loc[i, "Artist"], self.df_songs.loc[i, "Song_name"])

           batch_lyrics= self.df_songs.loc[i, "Lyrics"]
           batch_lyrics = re.sub("[\(\[].*?[\)\]]", "", batch_lyrics)

           token_batch_lyrics =  self.tokenizing_batch(batch_lyrics)
           token_batch_lyrics_data = token_batch_lyrics.data

           token_batch_lyrics_data['Lyrics'] = batch_lyrics
           #token_batch_lyrics_data['label']  =   self.df_songs.loc[i, "label"]


           self.songs_dict[key] = token_batch_lyrics_data


           print("debug")

           # self.df_songs.loc[i, "token_song_input_ids" ] =  token_song_data  ['input_ids']
           # self.df_songs.loc[i, "token_song_type_ids"] = token_song_data['token_type_ids']
           # self.df_songs.loc[i, "token_song_attention_mask"] = token_song_data['attention_mask']





