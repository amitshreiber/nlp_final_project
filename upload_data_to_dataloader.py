import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class upload_data_to_dataloader:

    def __init__(self, orig_songs_df, data, args):
        self.labels = orig_songs_df['label'].values
        self.data = data
        self.args = args

        self.create_dataloader()

    def create_dataloader(self):
        train_features, test_features, train_labels, test_labels = train_test_split(self.data, self.labels,
                                                                                    test_size= self.args.test_ratio,
                                                                                    random_state=42)

        if self.args.validation_ratio> 0:
            train_features, val_features, train_labels, val_labels = train_test_split(train_features,
                                                                                      train_labels,
                                                                                      test_size= self.args.validation_ratio,
                                                                                      random_state=42
                                                                                      )

        tr_dataset = TensorDataset(train_features, torch.from_numpy(train_labels))
        self.tr_dataloader = DataLoader(tr_dataset, batch_size=self.args.tr_batch_size)

        if self.args.validation_ratio> 0:
            val_dataset = TensorDataset(val_features, torch.from_numpy(val_labels))
            self.val_dataloader = DataLoader(val_dataset, batch_size= self.args.val_batch_size)
