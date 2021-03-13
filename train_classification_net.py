import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from handy_function import print_current_time
import torch




class train_classification_net():


        def __init__(self, train_dataloader, net, args,  optimizer, device ):


            self.train_dataloader = train_dataloader
            #self.val_dataloader = val_dataloader
            self.epochs = args.num_epochs
            self.optimizer = optimizer
            self.net = net.to(device)
            self.device = device
            #self.use_validation = True
            self.criterion = args.criterion


            self.train_loss= [None] * self.epochs
            self.val_loss= [None] * self.epochs


            self.train_net()


        def train_net(self):

             first_batch_train = True

             print_current_time("starting to train classifier net")

             for epoch in range(self.epochs):

                    epoch_total_train_loss = 0.0

                    self.net.train()

                    for x_train,y_train in tqdm(self.train_dataloader):
                            x_train = x_train.to(self.device)
                            y_train = y_train.to(self.device)



                            self.optimizer.zero_grad()

                            y_train = y_train.squeeze_()
                            #y_train = torch.transpose(y_train, 0, 1)

                            y_pred = self.net(x_train)

                            loss =   self.criterion (y_pred, y_train.long())
                            loss.backward()

                            epoch_total_train_loss += loss

                            if first_batch_train:

                                first_batch_train = False
                                print ("\nThe loss value for the first batch in the first epoch: ", loss.item(), "\n")

                            self.optimizer.step()


                    self.update_train_metrics(epoch, epoch_total_train_loss )
                    self.print_metrics(epoch, Train = True)


        def update_train_metrics(self, epoch, epoch_total_train_loss):
             self.train_loss[epoch] = epoch_total_train_loss.item() / len(self.train_dataloader)


        def print_metrics(self, epoch, Train):
                if epoch % 10 == 0:
                    print()
                    print("******************************")

                    if Train:
                        print_current_time("Training metrics for epoch " + str(epoch) + ":\n")
                        print("Loss value: ", self.train_loss[epoch])

                    # else:
                    #     self.epoch_before_eraly_stop = epoch
                    #     print_current_time("Validation metrics for epoch " + str(epoch) + ":\n")
                    #     print("Correct unifrom masked ones ratio: ", self.val_correct_unifrom_masked_ones_ratio[epoch])
                    #     print("Correct popularity masked ones ratio: ", self.val_correct_pop_masked_ones_ratio[epoch])
                    #     print("Loss value: ", self.val_loss[epoch])



