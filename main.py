#main
import torch
from Tokenizing import *
from embedding import *
from args import *
import os
from classification_net import *
from upload_embedding_to_dataloader import *
from train_classification_net import *
from torch import optim
# from train_classification_net import *
# from torch.utils.data import DataLoader
# from torch.utils.data import TensorDataset

print("conda environment:", os.environ['CONDA_DEFAULT_ENV'], "\n \n")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

args = args()


song_token = tokenizing (texts_csv_path= "bert toy example.csv" )
song_token.tokenize_each_song()
print("debug")
embedding_songs = embedding(tokenizing_data= song_token.songs_dict, device= device)
embedding_songs.data_embedding()
embedding_dataloaders = upload_embedding_to_dataloader(song_token.df_songs, embedding_songs.songs_features, args )

print ("debug")


classification_net= classification_net(args)


if args.weight_decay> 0 :
  adam_optimizer = optim.Adam(classification_net.parameters(), lr=args.lr , weight_decay= args.weight_decay )
else:
    adam_optimizer = optim.Adam(classification_net.parameters(), lr=args.lr)


training_net =  train_classification_net(train_dataloader= embedding_dataloaders.tr_dataloader, optimizer= adam_optimizer, device= device, net= classification_net, args= args)

