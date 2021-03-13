import torch
import pandas as pd
import numpy as np
import datetime
import torch.nn.functional



class args:

    def __init__(self,  hidden_dim = 50, validation_ratio = 0, test_ratio = 0.0001,
                  tr_batch_size =3, val_batch_size = 128, num_epochs= 10,
                  lr = 0.0005, weight_decay= 1e-4 ):






        # Train-test split args

         self.test_ratio = test_ratio
         self.validation_ratio = validation_ratio

        # NN architecture args
         self.hidden_dim =  hidden_dim

        # NN training args

         self.tr_batch_size =  tr_batch_size
         self.val_batch_size = val_batch_size
         self.num_epochs = num_epochs


    # NN backpropagation args

         self.lr = lr
         self.weight_decay = weight_decay
         self.criterion =  torch.nn.CrossEntropyLoss()

    ## classes
         self.class_number = 19












