import torch
from handy_function import print_current_time
from pytorch_transformers import BertModel


class Embedding:
    def __init__(self, tokenizing_data, device, embedding_csv_path=None):
        # Load pretrained model/tokenizer
        self.device = device
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.tokenizing_data = tokenizing_data
        self.embedding_csv_path = embedding_csv_path
        self.songs_features = torch.tensor([])

    def data_embedding(self):
        if self.embedding_csv_path is not None:
            self.songs_features = torch.load(self.embedding_csv_path)
        else:
            self.songs_features = torch.tensor([]).to(self.device)
            self.bert.model = self.bert.model.to(self.device)
            self.bert.model.eval()

            # self.training_data = self.training_data.to(self.device)
            i = 0
            print_current_time("starting embedding the data using Bert")

            for key, token_song in self.tokenizing_data.items():
                with torch.no_grad():
                    input_ids = token_song.get('input_ids').to(self.device)
                    attention_mask = token_song.get('attention_mask').to(self.device)
                    y_embedding = self.bert.model(input_ids=input_ids, token_type_ids=None,
                                                  attention_mask=attention_mask)
                    song_features = y_embedding[0][:, 0, :]
                    self.songs_features = torch.cat((self.songs_features, song_features), dim=0)
                    if i % 20 == 0:
                        print_current_time("embed " + str(i))
                    i += 1

            print_current_time("finish to embed the data using Bert")
