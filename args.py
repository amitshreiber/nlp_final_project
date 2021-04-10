import torch.nn.functional


class args:

  def __init__(self):

    #NN architecture args
    self.input_size = 768
    self.class_number= 19
    self.p1= 0.0
    self.p2 = 0.0
    self.fc1_output_size =  512
    self.fc2_output_size = 64


    # Train-test split args
    self.validation_ratio = 0.2
    self.test_ratio = 0.2

    # NN training args
    self.tr_batch_size =  128
    self.val_batch_size = 128
    self.test_batch_size = 128
    self.num_epochs = 40

    # NN back propagation args
    self.lr = 0.003
    self.weight_decay= 0.0001
    self.criterion = torch.nn.CrossEntropyLoss()

    # NN early stopping args
    self.early_stop_n = 10
    self.early_stop_acc_value = 0


