import torch
from Tokenizing import Tokenizing
from embedding import Embedding
import os
from classification_net import ClassificationNet
from bert_include_classifer_net import BERTClassifer
from upload_data_to_dataloader import upload_data_to_dataloader
from train_net import TrainNet
from preprocess_data import PreprocessData
from torch import optim
from plots import plot_accuracies, plot_loss
from directories import ROOT_DIR, PARAMETERS_DIR
from args import args

tr_bert_classifer =  True

# set device to GPU
print("conda environment:", os.environ['CONDA_DEFAULT_ENV'], "\n \n")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#args
args = args()

# pre-process data
data_file = os.path.join(ROOT_DIR, r'data\all_songs_nineteen_artists.csv')
songs_df = PreprocessData(data_file, 512)

# tokenize songs
tokenizing_path = os.path.join(PARAMETERS_DIR, "all_songs_token.pt")
song_token = Tokenizing(df_songs= songs_df.filtered_df_songs )
song_token.tokenize_each_song(tokenizing_path)

if  tokenizing_path is None:
    torch.save(song_token.songs_dict, PARAMETERS_DIR + "\\all_songs_token.pt" )

if not tr_bert_classifer:
    # song embeddings
    embedding_path = os.path.join(PARAMETERS_DIR, "embedding_all_artist.pt")
    embedding_songs = Embedding(tokenizing_data=song_token.songs_dict, device=device, embedding_path= embedding_path)
    embedding_songs.data_embedding()

    if embedding_songs.embedding_path is None:
        torch.save(embedding_songs.songs_features,  PARAMETERS_DIR + "\\embedding_all_artist.pt" )

    # create dataloader from embeddings
    embedding_dataloaders = upload_data_to_dataloader(song_token.df_songs, embedding_songs.songs_features, args= args)

else:
    embedding_dataloaders = upload_data_to_dataloader(song_token.df_songs, song_token.songs_dict, args=args, tokenized_data= True)

# train classification net
if tr_bert_classifer:

    net = BERTClassifer(args, device).model


else:
    net = ClassificationNet(args).to(device)

LR = args.lr
WEIGHT_DECAY = args.weight_decay

if WEIGHT_DECAY > 0:
    adam_optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
else:
    adam_optimizer = optim.Adam(net.parameters(), lr=LR)


training_net = TrainNet(train_dataloader=embedding_dataloaders.tr_dataloader, optimizer=adam_optimizer,
                                      device=device, net= net,
                                      val_dataloader=embedding_dataloaders.val_dataloader,
                                      args= args, tr_bert_classifer = tr_bert_classifer)

# plot figures
plot_accuracies(training_net.train_acc, training_net.val_acc, 'all_artists')
plot_loss(training_net.train_loss, training_net.val_loss, 'all_artists')

