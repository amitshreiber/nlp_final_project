import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

class upload_embedding_to_dataloader():


   def __init__(self, orig_songs_df, emmbeding_songs, args):

       self.labels = orig_songs_df['label'].values
       self.emmbeding_songs = emmbeding_songs
       self.test_ratio = args.test_ratio
       self.val_ratio = args.validation_ratio

       self.training_batch_size = args.tr_batch_size
       self.val_batch_size = args.val_batch_size


       self.create_dataloader()



   def create_dataloader(self):

       train_features, test_features, train_labels, test_labels = train_test_split(self.emmbeding_songs, self.labels,  test_size = self.test_ratio, random_state = 42)
       if self.val_ratio > 0:
        train_features, val_features, train_labels, val_labels = train_test_split(  train_features,                                                                                self.labels,
                                                                                    train_labels,
                                                                                    test_size= self.val_ratio,
                                                                                    random_state=42
                                                                                 )

       self.tr_dataloader= TensorDataset(train_features, torch.from_numpy(train_labels) )

       if self.val_ratio> 0:
           self.val_dataloader = TensorDataset(val_features, torch.from_numpy(val_labels))











