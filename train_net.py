# import pandas as pd
from tqdm import tqdm
from handy_function import print_current_time
import torch
import time
from handy_function import timeSince,save_model, calculate_accuracy



class TrainNet:
    def __init__(self, train_dataloader, net, optimizer, device, args, val_dataloader=None, save= False):

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.net = net.to(device)
        self.device = device
        self.save = save
        self.use_validation = args.validation_ratio  > 0
        self.num_epochs = args.num_epochs
        self.criterion = args.criterion
        self.early_stop_n = args.early_stop_n
        self.early_stop_acc_value = args.early_stop_acc_value
        self.epoch_before_early_stop = 0
        self.val_acc_value_before_eraly_stop = 0.0
        self.val_loss_value_before_eraly_stop = 0.0

      
        



        self.train_loss = [None] *  self.num_epochs
        self.val_loss = [None] *  self.num_epochs

        self.train_acc = [None] *  self.num_epochs
        self.val_acc = [None] *  self.num_epochs

        self.train_net()

    def train_net(self):

        start = time.time()
        print_current_time("starting to train classifier net")
        epoch_total_train_loss = 0.0  # Reset every epoch

        for epoch in range( self.num_epochs):
            for x_train, y_train in tqdm(self.train_dataloader):
                self.net.train()
                x_train, y_train = x_train.to(self.device), y_train.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                y_train = y_train.squeeze_()
                y_pred = self.net(x_train)
                loss =  self.criterion(y_pred, y_train.long())
                loss.backward()
                self.optimizer.step()

                epoch_total_train_loss += loss.item()

            # train metrics
            self.train_loss[epoch] = epoch_total_train_loss / len(self.train_dataloader)
            epoch_total_train_loss = 0.0

            self.train_acc[epoch], _ = self.evaluate(self.train_dataloader)

            if self.use_validation:
                # val metrics
                self.val_acc[epoch], self.val_loss[epoch] = self.evaluate(self.val_dataloader, True)
                self.print_metrics(epoch, start, False )


                self.epoch_before_early_stop = epoch
                self.val_acc_value_before_eraly_stop =   self.val_acc[epoch]
                self.val_loss_value_before_eraly_stop =  self.val_loss[epoch]
                #early stop check
                if self.early_stopping_check(epoch):
                        break



            self.print_metrics(epoch, start)
            # print(f'Epoch #{epoch}:\n'
            #       f'Last batch Loss: {loss.item():.4f}\n'
            #       f'Train accuracy: {epoch_total_train_acc:.3f}\n'
            #       f'Test accuracy: {test_accuracy:.3f}\n'
            #       f'Time elapsed (remaining): {timeSince(start, (epoch+1) /  self.num_epochs)}')

            if self.save:
                save_model(self.net, epoch)



    def print_metrics(self, epoch, start, train=True):
        if epoch % 20 == 0:
            print()
            print("******************************")

            if train:
                print(f'Epoch #{epoch+1}:\n'
                      f'Train Loss: {self.train_loss[epoch]:.4f}\n'
                      f'Train accuracy: {self.train_acc[epoch]:.3f}\n'
                      f'Time elapsed (remaining): {timeSince(start, (epoch+1) /  self.num_epochs)}')

            else:

                print(f'Epoch #{epoch + 1}:\n'
                      f'Validation Loss: {self.val_loss[epoch]:.4f}\n'
                      f'Validation accuracy: {self.val_acc[epoch]:.3f}\n' )

    def early_stopping_check(self, curr_epoch):
        if curr_epoch <  self.early_stop_n :
            return False



        for i in range(0,  self.early_stop_n):
            if self.val_acc[curr_epoch - i] - self.val_acc[curr_epoch - i - 1] >= self.early_stop_acc_value:
                return False


        print("made early stopping after epoch: ", curr_epoch)
        return True

    def evaluate(self, dataloader, val = False):

        total = 0.0
        correct = 0.0
        epoch_val_loss = 0.0
        self.net.eval()

        with torch.no_grad():

            for inputs, labels in tqdm(dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.squeeze_()
                outputs = self.net(inputs)

                current_correct, current_total = calculate_accuracy(outputs, labels)
                correct += current_correct
                total += current_total

                if val:
                    val_loss = self.criterion(outputs, labels.long())
                    epoch_val_loss += val_loss

        accuracy = correct / total
        epoch_val_avg_loss = epoch_val_loss/len(self.train_dataloader)

        return accuracy, epoch_val_avg_loss











