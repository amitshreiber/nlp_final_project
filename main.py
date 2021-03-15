# main
import torch
from Tokenizing import Tokenizing
from embedding import Embedding
from args import args
import os
from classification_net import ClassificationNet
from upload_embedding_to_dataloader import upload_embedding_to_dataloader
from train_classification_net import TrainClassificationNet
from preprocess_data import PreprocessData
from torch import optim
# from functions_utils import
# from train_classification_net import *
# from torch.utils.data import DataLoader
# from torch.utils.data import TensorDataset
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))  # This is your Project Root
PARAMETERS_DIR = os.path.join(ROOT_DIR, 'parameters')

# set device to GPU
print("conda environment:", os.environ['CONDA_DEFAULT_ENV'], "\n \n")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#
args = args()
data_file = os.path.join(ROOT_DIR, r'data\just_3_artist.csv')
songs_df = PreprocessData(data_file, 512)
song_token = Tokenizing(songs_df.filtered_df_songs)
song_token.tokenize_each_song()

embedding_songs = Embedding(tokenizing_data=song_token.songs_dict, device=device)
embedding_songs.data_embedding()
if embedding_songs.embedding_csv_path is None:
    torch.save(embedding_songs.songs_features,  os.path.join(PARAMETERS_DIR, "embedding_3_artist.pt"))

embedding_dataloaders = upload_embedding_to_dataloader(song_token.df_songs, embedding_songs.songs_features, args )

classification_net = ClassificationNet(args).to(device)


if args.weight_decay > 0:
    adam_optimizer = optim.Adam(classification_net.parameters(), lr=args.lr , weight_decay= args.weight_decay )
else:
    adam_optimizer = optim.Adam(classification_net.parameters(), lr=args.lr)

training_net = TrainClassificationNet(train_dataloader=embedding_dataloaders.tr_dataloader, optimizer=adam_optimizer,
                                      device=device,
                                      net=classification_net, args=args,
                                      val_dataloader=embedding_dataloaders.val_dataloader)
