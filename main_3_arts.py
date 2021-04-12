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
import pickle
import copy

tr_bert_classifer =  False

# set device to GPU
print("conda environment:", os.environ['CONDA_DEFAULT_ENV'], "\n \n")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#args
args = args()
args.class_number = 3


# pre-process data
data_file = os.path.join(ROOT_DIR, r'data\just_3_artist.csv')
songs_df = PreprocessData(data_file, 512)

# tokenize songs
tokenizing_path = os.path.join(PARAMETERS_DIR, "songs_five_artists_token.pt")
song_token = Tokenizing(df_songs= songs_df.filtered_df_songs )
song_token.tokenize_each_song(tokenizing_path)

if  tokenizing_path is None:
    torch.save(song_token.songs_dict, PARAMETERS_DIR + "\\songs_five_artists_token.pt" )

if not tr_bert_classifer:
    # song embeddings
    embedding_path = os.path.join(PARAMETERS_DIR, "embedding_3_artist.pt")
    embedding_songs = Embedding(tokenizing_data=song_token.songs_dict, device=device, embedding_path= embedding_path)
    embedding_songs.data_embedding()

    if embedding_songs.embedding_path is None:
        torch.save(embedding_songs.songs_features,  PARAMETERS_DIR + "\\embedding_3_artist.pt" )

    # create dataloader from embeddings
    embedding_dataloaders = upload_data_to_dataloader(song_token.df_songs, embedding_songs.songs_features, args= args)

else:
    embedding_dataloaders = upload_data_to_dataloader(song_token.df_songs, song_token.songs_dict, args=args, tokenized_data= True)

# train classification net
if tr_bert_classifer:

    bert_classifer = BERTClassifer(args, device)
    untrained_net = bert_classifer.model


else:
    untrained_net = ClassificationNet(args, input_size=args.input_size).to(device)

untrained_net_final = copy.deepcopy(untrained_net)


LR = args.lr
WEIGHT_DECAY = args.weight_decay

if WEIGHT_DECAY > 0:
    adam_optimizer = optim.Adam(untrained_net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
else:
    adam_optimizer = optim.Adam(untrained_net.parameters(), lr=LR)


trained_net = TrainNet(train_dataloader=embedding_dataloaders.tr_dataloader, optimizer=adam_optimizer,
                                      device=device, net=    untrained_net,
                                      val_dataloader=embedding_dataloaders.val_dataloader,
                                      args= args, tr_bert_classifer = tr_bert_classifer, k =2)

model_result = {'loss': trained_net.train_loss, 'train_acc': trained_net.train_acc, 'test_acc': trained_net.val_acc}


print("best validation accuracy was: ", round(trained_net.val_best_acc_value, 4), "after epoch number: ", trained_net.val_best_acc_epoch)


#plot figures
plot_accuracies(trained_net.train_acc, trained_net.val_acc, 'all_artists')
plot_loss(trained_net.train_loss, trained_net.val_loss, 'all_artists')



with open(os.path.join(PARAMETERS_DIR, 'bert_not_include_classifer.pkl'), 'wb') as f:
    pickle.dump(model_result, f)





print("training final net")


args.validation_ratio = 0
args.early_stop_n = 1000000
args.num_epochs = trained_net.val_best_acc_epoch

del trained_net



if not tr_bert_classifer:
    embedding_dataloaders_without_val = upload_data_to_dataloader(song_token.df_songs, embedding_songs.songs_features, args= args)


else:
    embedding_dataloaders_without_val = upload_data_to_dataloader(song_token.df_songs, song_token.songs_dict, args=args, tokenized_data= True)


if WEIGHT_DECAY > 0:
    adam_optimizer = optim.Adam(untrained_net_final.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
else:
    adam_optimizer = optim.Adam(untrained_net_final.parameters(), lr=LR)



final_net = TrainNet(train_dataloader= embedding_dataloaders_without_val.tr_dataloader, optimizer=adam_optimizer,
                                      device=device, net= untrained_net_final,
                                      val_dataloader=embedding_dataloaders_without_val.test_dataloader,
                                      args= args, tr_bert_classifer = tr_bert_classifer, k = 2)




