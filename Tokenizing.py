from transformers import BertTokenizer
from handy_function import print_current_time


class Tokenizing:
    def __init__(self, df_songs):
        self.df_songs = df_songs
        self.max_embed_batch_len = 512
        self.songs_dict = {}
        self.create_tokenizer_instance()
        self.tokenizer = None

    def create_tokenizer_instance(self):
        # Load the BERT tokenizer.
        print('Loading BERT tokenizer...')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def tokenizing_batch (self, batch):
        """
        For every single song

        """

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
            max_length=self.max_embed_batch_len,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt'  # Return pytorch tensors.
        )
        return encoded_dict

    def tokenize_each_song(self):
        """
        For all songs

        """
        print_current_time("starting tokenizing process")

        for i in range(len(self.df_songs)):
            key = (self.df_songs.loc[i, "Artist"], self.df_songs.loc[i, "Song_name"])

            batch_lyrics = self.df_songs.loc[i, "Lyrics"]

            token_batch_lyrics = self.tokenizing_batch(batch_lyrics)
            token_batch_lyrics_data = token_batch_lyrics.data

            token_batch_lyrics_data['Lyrics'] = batch_lyrics

            self.songs_dict[key] = token_batch_lyrics_data

        print_current_time("finished tokenizing process")
