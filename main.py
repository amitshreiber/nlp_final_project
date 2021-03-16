import torch
from Tokenizing import Tokenizing
from embedding import Embedding
import os
from classification_net import ClassificationNet
from upload_embedding_to_dataloader import upload_embedding_to_dataloader
from train_classification_net import TrainClassificationNet
from preprocess_data import PreprocessData
from torch import optim
from handy_function import plot_accuracies, plot_loss
from args import ROOT_DIR, PARAMETERS_DIR, WEIGHT_DECAY, LR

# set device to GPU
print("conda environment:", os.environ['CONDA_DEFAULT_ENV'], "\n \n")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# pre-process data
data_file = os.path.join(ROOT_DIR, r'data\songs_five_artists.csv')
songs_df = PreprocessData(data_file, 512)

# tokenize songs
tokenizing_path = os.path.join(PARAMETERS_DIR, "songs_five_artists_token.pt")
song_token = Tokenizing(songs_df.filtered_df_songs)
song_token.tokenize_each_song()
if song_token.tokenizing_path is None:
    torch.save(song_token.songs_dict, tokenizing_path)

# song embeddings
embedding_path = os.path.join(PARAMETERS_DIR, "embedding_songs_five_artists.pt")
embedding_songs = Embedding(tokenizing_data=song_token.songs_dict, device=device)
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
                                      val_dataloader=embedding_dataloaders.val_dataloader)

# plot figures
plot_accuracies(training_net.train_acc, training_net.val_acc, 'classification_five_artists')
plot_loss(training_net.train_loss, 'classification_five_artists')
b = 1
