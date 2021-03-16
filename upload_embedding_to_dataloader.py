import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from args import TEST_RATIO, VALIDATION_RATIO, TR_BATCH_SIZE, VAL_BATCH_SIZE


class upload_embedding_to_dataloader:
    def __init__(self, orig_songs_df, emmbeding_songs):
        self.labels = orig_songs_df['label'].values
        self.emmbeding_songs = emmbeding_songs
        self.create_dataloader()

    def create_dataloader(self):
        train_features, test_features, train_labels, test_labels = train_test_split(self.emmbeding_songs, self.labels,
                                                                                    test_size=TEST_RATIO,
                                                                                    random_state=42)

        if VALIDATION_RATIO > 0:
            train_features, val_features, train_labels, val_labels = train_test_split(train_features,
                                                                                      train_labels,
                                                                                      test_size=VALIDATION_RATIO,
                                                                                      random_state=42
                                                                                      )

        tr_dataset = TensorDataset(train_features, torch.from_numpy(train_labels))
        self.tr_dataloader = DataLoader(tr_dataset, batch_size=TR_BATCH_SIZE)

        if VALIDATION_RATIO > 0:
            val_dataset = TensorDataset(val_features, torch.from_numpy(val_labels))
            self.val_dataloader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE)
