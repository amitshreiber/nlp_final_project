import torch
import torch.nn.functional


class args:
    def __init__(self, hidden_dim=50, validation_ratio=0.2, test_ratio=0.2,
              tr_batch_size=256, val_batch_size=256, num_epochs=1000,
              lr=0.001, weight_decay=0):

        # Train-test split args
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio

        # NN architecture args
        self.hidden_dim = hidden_dim

        # NN training args
        self.tr_batch_size = tr_batch_size
        self.val_batch_size = val_batch_size
        self.num_epochs = num_epochs

        # NN backpropagation args
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = torch.nn.CrossEntropyLoss()

        # NN early stopping args
        self.early_stop_n = 3
        self.early_stop_loss_value = 0

        # classes
        self.class_number = 19
