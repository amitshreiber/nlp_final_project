import torch
from Tokenizing import Tokenizing
from embedding import Embedding
import os
from upload_data_to_dataloader import upload_data_to_dataloader
from preprocess_data import PreprocessData
from plots import plot_accuracies, plot_loss
from args_CV import ROOT_DIR, PARAMETERS_DIR
from args_comb import ArgsComb
from trained_classification_nets_parameters import   TrainedClassificationNetsParams



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

# song embeddings
embedding_path = os.path.join(PARAMETERS_DIR, "embedding_all_artist.pt")
embedding_songs = Embedding(tokenizing_data=song_token.songs_dict, device=device, embedding_path= embedding_path)

embedding_songs.data_embedding()

if embedding_songs.embedding_path is None:
    torch.save(embedding_songs.songs_features,  PARAMETERS_DIR + "\\embedding_all_artist.pt" )

classification_nets_params = TrainedClassificationNetsParams()

index = 0
args_comb = ArgsComb()
for args in args_comb.args_combs_list:
   if index > 8649 and args.tr_batch_size<128:

    embedding_dataloaders = upload_data_to_dataloader (song_token.df_songs, embedding_songs.songs_features, args)



# train classification net
    classification_nets_params.train_net(device, args, embedding_dataloaders, index)
   index += 1
   if index % 50 == 0 and index > 8649 and args.tr_batch_size<128:
        classification_nets_params.parms_to_csv()
        print(index)





classification_nets_params.parms_to_csv()
print("finish")


# plot figures
# plot_accuracies(training_net.train_acc, training_net.val_acc, 'all_artists')
# plot_loss(training_net.train_loss, training_net.val_loss, 'all_artists')

