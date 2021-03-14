import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from handy_function import print_current_time
import torch




class train_classification_net():


        def __init__(self, train_dataloader, net, args,  optimizer, device, val_dataloader = None ):


            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader
            self.epochs = args.num_epochs
            self.optimizer = optimizer
            self.net = net.to(device)
            self.device = device
            #self.use_validation = True
            self.criterion = args.criterion

            self.early_stop_n = args.early_stop_n
            self.early_stop_loss_value =  args.early_stop_loss_value

            self.use_validation = False
            if args.validation_ratio > 0:
                self.use_validation = True


            self.train_loss= [None] * self.epochs
            self.val_loss= [None] * self.epochs

            self.train_acc = [None] * self.epochs
            self.val_acc =   [None] * self.epochs


            self.train_net()


        def train_net(self):

             first_batch_train = True

             print_current_time("starting to train classifier net")

             for epoch in range(self.epochs):

                    epoch_total_train_loss = 0.0
                    epoch_total_train_acc = 0.0

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

                            epoch_total_train_acc +=   self.calculate_accuracy(y_pred, y_train)

                            if first_batch_train:

                                first_batch_train = False
                                print ("\nThe loss value for the first batch in the first epoch: ", loss.item(), "\n")

                            self.optimizer.step()


                    self.update_train_metrics(epoch, epoch_total_train_loss, epoch_total_train_acc )
                    self.print_metrics(epoch, Train = True)

                    if self.use_validation:
                         self.check_validation(epoch)

                         if self.early_stopping_check(epoch):
                             break




        def update_train_metrics(self, epoch, epoch_total_train_loss,  epoch_total_train_acc):

             self.train_loss[epoch] = epoch_total_train_loss.item() / len(self.train_dataloader)
             self.train_acc[epoch]  =  epoch_total_train_acc / len(self.train_dataloader)

        def print_metrics(self, epoch, Train):
                if epoch % 20 == 0:
                    print()
                    print("******************************")

                    if Train:
                        print_current_time("Training metrics for epoch " + str(epoch) + ":\n")
                        print("Loss value: ", self.train_loss[epoch])
                        print("accuracy value: ", self.train_acc[epoch])

                    else:
                        self.epoch_before_eraly_stop = epoch
                        print_current_time("Validation metrics for epoch " + str(epoch) + ":\n")
                        print("Loss value: ", self.val_loss[epoch])
                        print("accuracy value: ", self.val_acc[epoch])



        def early_stopping_check(self, curr_epoch):

            if curr_epoch <  self.early_stop_n -1:
                return False

            loss_improved = False

            for i in range(0,self.early_stop_n):
                if self.val_loss[curr_epoch-i ] -  self.val_loss[curr_epoch - i -1] <= self.early_stop_loss_value:
                    loss_improved = True
                    return False

            if not loss_improved:
                print("made early stopping after epoch: ", curr_epoch)
                return True

        def check_validation(self, epoch):

            self.val_loss[epoch], self.val_acc[epoch] = self.get_val_metrics()
            self.print_metrics(epoch, Train=False)





        def get_val_metrics(self):

            epoch_total_val_acc = 0.0
            epoch_total_val_loss = 0.0
            val_dataloader_length = len(self.val_dataloader)

            self.net.eval()

            for x_val, y_val in tqdm(self.val_dataloader):

                x_val = x_val.to(self.device)
                y_val = y_val.to(self.device)
                y_val = y_val.squeeze_()

                with torch.no_grad():
                    y_val_predicted = self.net(x_val)
                    loss = self.criterion(y_val_predicted ,y_val.long())
                    epoch_total_val_loss += loss

                    epoch_total_val_acc += self.calculate_accuracy( y_val_predicted, y_val)

            epoch_val_loss = epoch_total_val_loss.item() / val_dataloader_length
            epoch_val_acc =  epoch_total_val_acc / val_dataloader_length

            return   epoch_val_loss,  epoch_val_acc

        def calculate_accuracy(self, y_pred, y_train):
            pass









