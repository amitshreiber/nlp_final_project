import pandas as pd
from classification_net import ClassificationNet
from preprocess_data import PreprocessData
import os
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from directories import ROOT_DIR, PARAMETERS_DIR
import torch
from args import args
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from train_net import TrainNet
from plots import plot_accuracies, plot_loss
import pickle
np.random.seed(2018)


def convert_sparse_matrix_to_sparse_tensor(x):
    coo = x.tocoo()
    indices = np.mat([coo.row, coo.col])
    return torch.sparse_coo_tensor(indices, coo.data, coo.shape)


def preprocess_text(text):
    # Tokenize words while ignoring punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    # Lowercase and lemmatise
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token.lower(), pos='v') for token in tokens]

    # Remove stopwords
    keywords = [lemma for lemma in lemmas if lemma not in stopwords.words('english')]
    return keywords


def upload_data_to_dataloader(train_features, train_labels, val_features, val_labels, test_features, test_labels, args):
    tr_dataset = TensorDataset(train_features, train_labels)
    tr_dataloader = DataLoader(tr_dataset, batch_size=args.tr_batch_size)

    val_dataset = TensorDataset(val_features, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size)

    test_dataset = TensorDataset(test_features, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=args.tr_batch_size)

    return tr_dataloader, val_dataloader, test_dataloader


def tfiddf_embeddings(args, songs_df, embedding_path=None):
    # initialize
    tfidf_embedding_val = torch.tensor([], dtype=torch.float64)
    tfidf_embedding_train = torch.tensor([], dtype=torch.float64)
    tfidf_embedding_test = torch.tensor([], dtype=torch.float64)
    train_labels = torch.tensor([])
    val_labels = torch.tensor([])
    test_labels = torch.tensor([])

    nltk.download('wordnet')
    nltk.download('stopwords')

    # if saved tokenizing
    if embedding_path is not None:
        dirname = os.path.dirname(embedding_path)
        basename = os.path.basename(embedding_path)
        tfidf_embedding_train = torch.load(os.path.join(dirname, f'train_{basename}')).float()
        train_labels = torch.load(os.path.join(dirname, f'train_labels.pt'))
        tfidf_embedding_test = torch.load(os.path.join(dirname, f'test_{basename}')).float()
        test_labels = torch.load(os.path.join(dirname, f'test_labels.pt'))

        # with validation
        if args.validation_ratio > 0:
            tfidf_embedding_val = torch.load(os.path.join(dirname, f'val_{basename}')).float()
            val_labels = torch.load(os.path.join(dirname, f'val_labels.pt'))

    else:
        # Create an instance of TfidfVectorizer
        vectoriser = TfidfVectorizer(analyzer=preprocess_text)

        # split data
        train_features, test_features, train_labels, test_labels = train_test_split(songs_df.df_songs.Lyrics,
                                                                                    songs_df.df_songs.label,
                                                                                    test_size=args.test_ratio,
                                                                                    random_state=42)
        # Fit to the data and transform to feature matrix
        tfidf_embedding_train = vectoriser.fit_transform(train_features)

        if args.validation_ratio > 0:
            train_features, val_features, train_labels, val_labels = train_test_split(train_features,
                                                                                      train_labels,
                                                                                      test_size=args.validation_ratio,
                                                                                      random_state=42)
            # Fit to the data and transform to feature matrix
            tfidf_embedding_train = vectoriser.fit_transform(train_features)
            tfidf_embedding_val = convert_sparse_matrix_to_sparse_tensor(vectoriser.transform(val_features)).to_dense()
            torch.save(tfidf_embedding_val, PARAMETERS_DIR + "\\val_tfidf.pt")
            torch.save(torch.tensor(val_labels.values), PARAMETERS_DIR + r"\val_labels.pt")

        # transform the test data
        tfidf_embedding_test = convert_sparse_matrix_to_sparse_tensor(vectoriser.transform(test_features)).to_dense()

        # save params
        tfidf_embedding_train = convert_sparse_matrix_to_sparse_tensor(tfidf_embedding_train).to_dense()
        torch.save(tfidf_embedding_train, PARAMETERS_DIR + "\\train_tfidf.pt")
        torch.save(torch.tensor(train_labels.values), PARAMETERS_DIR + r"\train_labels.pt")
        torch.save(tfidf_embedding_test, PARAMETERS_DIR + "\\test_tfidf.pt")
        torch.save(torch.tensor(test_labels.values), PARAMETERS_DIR + r"\test_labels.pt")

    # upload to dataloader
    tr_dataloader, val_dataloader, test_dataloader = upload_data_to_dataloader(tfidf_embedding_train, train_labels,
                                                                               tfidf_embedding_val, val_labels,
                                                                               tfidf_embedding_test, test_labels,
                                                                               args)
    return tr_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':

    # set device to GPU
    print("conda environment:", os.environ['CONDA_DEFAULT_ENV'], "\n \n")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # args
    args = args()

    # pre-process data
    data_file = os.path.join(ROOT_DIR, r'data\all_songs_nineteen_artists.csv')
    songs_df = PreprocessData(data_file, 512)

    # TF-IDF
    embedding_path = os.path.join(PARAMETERS_DIR, "tfidf.pt")
    tr_dataloader, val_dataloader, test_dataloader = tfiddf_embeddings(args, songs_df, embedding_path=embedding_path)

    # classification
    net = ClassificationNet(args, input_size=tr_dataloader.dataset.tensors[0].size(1)).to(device)

    lr = args.lr
    WEIGHT_DECAY = args.weight_decay

    if WEIGHT_DECAY > 0:
        adam_optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    else:
        adam_optimizer = optim.Adam(net.parameters(), lr=lr)

    training_net = TrainNet(train_dataloader=tr_dataloader, optimizer=adam_optimizer, device=device, net=net,
                            val_dataloader=test_dataloader, args=args)

    model_result = {'loss': training_net.train_loss, 'train_acc': training_net.train_acc, 'test_acc': training_net.val_acc}

    with open(os.path.join(PARAMETERS_DIR, 'tfidf.pkl'), 'wb') as f:
        pickle.dump(model_result, f)

    model_result = pickle.load(open(os.path.join(PARAMETERS_DIR, 'x.pkl'), "rb"))
    # plot figures
    plot_accuracies(model_result['train_acc'], model_result['val_acc'], 'all_artists')
    plot_loss(training_net.train_loss, training_net.val_loss, 'all_artists')

