import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np




class upload_data_to_dataloader:

    def __init__(self, orig_songs_df, data, args,   tokenized_data = False ):

        self.labels = orig_songs_df['label'].values
        self.data = data
        self.args = args

        self.create_dataloader(tokenized_data)



    def create_dataloader(self,  tokenized_data):

       if  not tokenized_data:
            self.upload_emd_data()
       else:
            self.upload_tokenized_data()



    def upload_emd_data(self):

        train_features, test_features, train_labels, test_labels = train_test_split(self.data, self.labels,
                                                                                    test_size=self.args.test_ratio,
                                                                                    random_state=42)

        if self.args.validation_ratio > 0:
            train_features, val_features, train_labels, val_labels = train_test_split(train_features,
                                                                                      train_labels,
                                                                                      test_size=self.args.validation_ratio,
                                                                                      random_state=42
                                                                                      )
            val_dataset = TensorDataset(val_features, torch.from_numpy(val_labels))

            self.val_dataloader = DataLoader(val_dataset, batch_size=self.args.val_batch_size)

        tr_dataset = TensorDataset(train_features, torch.from_numpy(train_labels))

        self.tr_dataloader = DataLoader(tr_dataset, batch_size=self.args.tr_batch_size)

        test_dataset = TensorDataset(test_features, torch.from_numpy(test_labels))

        self.test_dataloader = DataLoader(test_dataset, batch_size=self.args.test_batch_size)



    def upload_tokenized_data(self):

        input_ids_data,  attention_mask_data  =   self.get_input_ids_att_maks_lists()

        train_input_id_data,  test_input_id_data,\
        train_attention_mask_data, test_attention_mask_data, \
        train_labels, test_labels = train_test_split(input_ids_data,  attention_mask_data, self.labels,test_size=self.args.test_ratio,   random_state=42)

        if self.args.validation_ratio > 0:
            train_input_id_data, val_input_id_data, \
            train_attention_mask_data, val_attention_mask_data, \
            train_labels, val_labels = train_test_split(train_input_id_data, train_attention_mask_data,  train_labels,
                                                         test_size=self.args.validation_ratio , random_state=42)

            val_dataset = TensorDataset(val_input_id_data, val_attention_mask_data ,torch.from_numpy(val_labels))

            self.val_dataloader = DataLoader(val_dataset, batch_size=self.args.val_batch_size)

        tr_dataset = TensorDataset(train_input_id_data, train_attention_mask_data ,torch.from_numpy(train_labels))

        self.tr_dataloader = DataLoader(tr_dataset, batch_size=self.args.tr_batch_size)

        test_dataset = TensorDataset(test_input_id_data, test_attention_mask_data ,torch.from_numpy(test_labels))

        self.test_dataloader =  DataLoader(test_dataset, batch_size=self.args.tr_batch_size)



    def get_input_ids_att_maks_lists(self):

        input_ids_data = torch.tensor([])
        attention_masks_data = torch.tensor([])

        for key, token_song in self.data.items():
                input_ids = token_song.get('input_ids')
                input_ids_data= torch.cat((input_ids_data,input_ids), dim = 0)

                attention_mask = token_song.get('attention_mask')
                attention_masks_data = torch.cat(( attention_masks_data, attention_mask), dim=0)


        return  input_ids_data, attention_masks_data




