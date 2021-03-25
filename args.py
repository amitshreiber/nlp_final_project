import torch.nn.functional


class args:

  def __init__(self):

    #NN architecture args
    self.hidden_dim = 50
    self.input_size = 768
    self.class_number= 19
    self.p1= 0.5
    self.p2 = 0.25
    self.fc1_output_size = 256
    self.fc2_output_size = 64


    # Train-test split args
    self.validation_ratio = 0.2
    self.test_ratio = 0.2

    # NN training args
    self.tr_batch_size =  512
    self.val_batch_size = 512
    self.num_epochs = 10000000

    # NN back propagation args
    self.lr = 0.0005
    self.weight_decay= 0.005
    self.criterion = torch.nn.CrossEntropyLoss()

    # NN early stopping args
    self.early_stop_n = 3
    self.early_stop_acc_value = 0


