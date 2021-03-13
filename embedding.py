from BERT_net import *
import torch

class embedding():


   def __init__(self, tokenizing_data, device):

       ## Load pretrained model/tokenizer
       self.device = device

       self.bert =  BERT_net()
       #self.bert_model.to(device)

       self.tokenizing_data = tokenizing_data





   def data_embedding(self):

       self.songs_features = torch.tensor([])

       self.bert.model.eval()
       #self.training_data = self.training_data.to(self.device)

       for key, token_song  in self.tokenizing_data.items():
           with torch.no_grad():
               input_ids  =token_song.get('input_ids')
               attention_mask =  token_song.get('attention_mask')
               y_embedding = self.bert.model(input_ids = input_ids, token_type_ids=None, attention_mask= attention_mask)
               song_features = y_embedding[0][:, 0, :]
               self.songs_features =torch.cat((self.songs_features, song_features), dim=0)






