import torch
from Tokenizing import Tokenizing
from embedding import Embedding
import os
from classification_net import ClassificationNet
from upload_embedding_to_dataloader import upload_embedding_to_dataloader
from train_classification_net import TrainClassificationNet
from preprocess_data import PreprocessData
from torch import optim
from plots import plot_accuracies, plot_loss
from args import ROOT_DIR, PARAMETERS_DIR, WEIGHT_DECAY, LR

# set device to GPU
print("conda environment:", os.environ['CONDA_DEFAULT_ENV'], "\n \n")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# pre-process data
data_file = os.path.join(ROOT_DIR, r'data\all_songs_nineteen_artists.csv')
songs_df = PreprocessData(data_file, 512)

# tokenize songs
tokenizing_path = os.path.join(PARAMETERS_DIR, "all_songs_token.pt")
song_token = Tokenizing(df_songs= songs_df.filtered_df_songs )
song_token.tokenize_each_song(tokenizing_path)

if song_token.tokenizing_path is None:
    torch.save(song_token.songs_dict, tokenizing_path)

# song embeddings
embedding_path = os.path.join(PARAMETERS_DIR, "embedding_all_artist_layer_11.pt")
embedding_songs = Embedding(tokenizing_data=song_token.songs_dict, device=device, embedding_path=embedding_path)
embedding_songs.data_embedding()
if embedding_songs.embedding_path is None:
    torch.save(embedding_songs.songs_features,  embedding_path)

# create dataloader from embeddings
embedding_dataloaders = upload_embedding_to_dataloader(song_token.df_songs, embedding_songs.songs_features)

# train classification net
classification_net = ClassificationNet().to(device)
if WEIGHT_DECAY > 0:
    adam_optimizer = optim.Adam(classification_net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
else:
    adam_optimizer = optim.Adam(classification_net.parameters(), lr=LR)


training_net = TrainClassificationNet(train_dataloader=embedding_dataloaders.tr_dataloader, optimizer=adam_optimizer,
                                      device=device, net=classification_net,
                                      val_dataloader=embedding_dataloaders.val_dataloader, k=3)

# plot figures
plot_accuracies(training_net.train_acc, training_net.val_acc, 'all_artists')
plot_accuracies(training_net.train_acc_k, training_net.val_acc_k, 'all_artists top-k')
plot_loss(training_net.train_loss, training_net.val_loss, 'all_artists')

