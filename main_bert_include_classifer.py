import torch
from Tokenizing import Tokenizing
from embedding import Embedding
import os
from upload_data_to_dataloader import upload_data_to_dataloader
from preprocess_data import PreprocessData
from plots import plot_accuracies, plot_loss
from args import ROOT_DIR, PARAMETERS_DIR
from args_comb import ArgsComb
from trained_classification_nets import   TrainedClassificationNets
from bert_include_classifer_net import *



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

if  tokenizing_path is None:
    torch.save(song_token.songs_dict, PARAMETERS_DIR + "\\all_songs_token.pt" )


token_data_dataloaders = upload_data_to_dataloader(args, song_token.df_songs, embedding_songs.songs_features) # change class name :upload_data

if args.weight_decay > 0:
    adam_optimizer = optim.Adam(classification_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
else:
    adam_optimizer = optim.Adam(classification_net.parameters(), lr=args.lr)


bert_include_classifer = BERTClassifer(args)

training_net = TrainNet(train_dataloader= token_data_dataloaders.tr_dataloader,
                                      optimizer=adam_optimizer,
                                      device=device, net= bert_include_classifer ,
                                      val_dataloader= token_data_dataloaders.val_dataloader,    #change name of class

                                      )


