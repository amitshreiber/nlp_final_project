import copy
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



print("conda environment:", os.environ['CONDA_DEFAULT_ENV'], "\n \n")



def tokenizing(songs_df, file_name):

    tokenizing_path = os.path.join(PARAMETERS_DIR, file_name)
    song_token = Tokenizing(df_songs= songs_df.filtered_df_songs)
    song_token.tokenize_each_song(tokenizing_path)

    if tokenizing_path is None:
        torch.save(song_token.songs_dict, PARAMETERS_DIR + "\\" +file_name)

    return song_token



def embedding(tr_bert_classier, file_name, song_tokens, device):

    if tr_bert_classier:
        return None

    # song embeddings
    embedding_path = os.path.join(PARAMETERS_DIR, file_name)
    embedding_songs = Embedding(tokenizing_data=song_tokens.songs_dict, device=device, embedding_path=embedding_path)
    embedding_songs.data_embedding()

    if embedding_songs.embedding_path is None:
        torch.save(embedding_songs.songs_features, PARAMETERS_DIR + "\\" + file_name)

    return  embedding_songs



def get_dataloaders(song_token, embedding_songs, args,tr_bert_classifer):

    if not tr_bert_classifer:

      embedding_dataloaders = upload_data_to_dataloader(song_token.df_songs, embedding_songs.songs_features, args=args)

    else:

     embedding_dataloaders = upload_data_to_dataloader(song_token.df_songs, song_token.songs_dict, args=args,
                                                  tokenized_data=True)

    return  embedding_dataloaders



def initialize_net(args, device, tr_bert_classifer):

    if tr_bert_classifer:

        bert_classifer = BERTClassifer(args, device)
        untrained_net = bert_classifer.model


    else:
        untrained_net = ClassificationNet(args, input_size=args.input_size).to(device)

    untrained_net_final = copy.deepcopy(untrained_net)

    return untrained_net, untrained_net_final



def get_adam_optimzer(net, args):

    LR = args.lr
    WEIGHT_DECAY = args.weight_decay

    if WEIGHT_DECAY > 0:
        adam_optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        adam_optimizer = optim.Adam(net.parameters(), lr=LR)

    return adam_optimizer



def train_net(dataloaders, adam_optimizer, device, untrained_net, args, tr_bert_classifer, predict_test = False):

    if not predict_test:
        val_dataloader = dataloaders.val_dataloader
    else:
        val_dataloader = dataloaders.test_dataloader


    trained_net = TrainNet(train_dataloader=dataloaders.tr_dataloader, optimizer=adam_optimizer,
                           device=device, net=untrained_net,
                           val_dataloader= val_dataloader,
                           args=args, tr_bert_classifer=tr_bert_classifer)

    model_result = {'loss': trained_net.train_loss, 'train_acc': trained_net.train_acc, 'test_acc': trained_net.val_acc}

    print("best validation accuracy was: ", round(trained_net.val_best_acc_value, 4), "after epoch number: ",
          trained_net.val_best_acc_epoch)

    with open(os.path.join(PARAMETERS_DIR, 'bert_not_include_classifer.pkl'), 'wb') as f:
        pickle.dump(model_result, f)

    return  trained_net, model_result



def update_args_test_predict(args):

    args.validation_ratio = 0
    args.early_stop_n = 1000000
    args.num_epochs = trained_net.val_best_acc_epoch

    return  args



#Main


tr_bert_classifer =  False

# set device to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#args
args = args()

# pre-process data
data_file = os.path.join(ROOT_DIR, r'data\all_songs_nineteen_artists.csv')
songs_df = PreprocessData(data_file, 512)

# tokenize songs
song_token = tokenizing(songs_df, "all_songs_token.pt")

#song embeddings
embedding_songs =  embedding(tr_bert_classifer, "embedding_all_artist.pt", song_token, device)

# upload data to dataloader
dataloaders = get_dataloaders(song_token, embedding_songs, args, tr_bert_classifer)

# initialize net
untrained_net, untrained_net_final =  initialize_net(args,device, tr_bert_classifer)

# get Adam optimzer
adam_optimizer = get_adam_optimzer(untrained_net, args)

# train_net
trained_net, model_results = train_net(dataloaders, adam_optimizer, device, untrained_net, args, tr_bert_classifer)

#plot figures
plot_accuracies(trained_net.train_acc, trained_net.val_acc, 'all_artists')
plot_loss(trained_net.train_loss, trained_net.val_loss, 'all_artists')


#training final net and predicting test results

print("training final net")

args= update_args_test_predict(args)

del trained_net, adam_optimizer, dataloaders

dataloaders_without_val = get_dataloaders(song_token, embedding_songs, args, tr_bert_classifer)

adam_optimizer = get_adam_optimzer(untrained_net_final, args)

trained_net, model_results = train_net(dataloaders_without_val , adam_optimizer, device, untrained_net_final, args, tr_bert_classifer, True)




