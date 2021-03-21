# import pandas as pd
from tqdm import tqdm
from handy_function import print_current_time
import torch
import time
from handy_function import timeSince,save_model, calculate_accuracy, top_k_accuracy
from args import CRITERION, EARLY_STOP_N, EARLY_STOP_ACC_VALUE, NUM_EPOCHS, VALIDATION_RATIO


class TrainClassificationNet:
    def __init__(self, train_dataloader, net, optimizer, device, val_dataloader=None, save=False, k=5, patience=-1):

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.net = net.to(device)
        self.device = device
        self.save = save
        self.use_validation = VALIDATION_RATIO > 0
        self.k = k
        self.patience = patience

        self.train_loss = [None] * NUM_EPOCHS
        self.val_loss = [None] * NUM_EPOCHS

        self.train_acc = [None] * NUM_EPOCHS
        self.train_acc_k = [None] * NUM_EPOCHS

        self.val_acc = [None] * NUM_EPOCHS
        self.val_acc_k = [None] * NUM_EPOCHS

        self.train_net()

    def train_net(self):

        start = time.time()
        print_current_time("starting to train classifier net")
        epoch_total_train_loss = 0.0  # Reset every epoch
        epochs_without_improvement = 0  # for early stopping (patience)
        best_dev_accuracy = 0

        for epoch in range(NUM_EPOCHS):
            for x_train, y_train in tqdm(self.train_dataloader):
                self.net.train()
                x_train, y_train = x_train.to(self.device), y_train.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                y_train = y_train.squeeze_()
                y_pred = self.net(x_train)
                loss = CRITERION(y_pred, y_train.long())
                loss.backward()
                self.optimizer.step()

                epoch_total_train_loss += loss.item()

            # train metrics
            self.train_loss[epoch] = epoch_total_train_loss / len(self.train_dataloader)
            epoch_total_train_loss = 0.0

            self.train_acc[epoch], self.train_acc_k[epoch], _ = self.evaluate(self.train_dataloader)

            if self.use_validation:
                # val metrics
                self.val_acc[epoch], self.val_acc_k[epoch], self.val_loss[epoch] = self.evaluate(self.val_dataloader, True)
                self.print_metrics(epoch, start, False)

                # early stop check
                if self.val_acc[epoch] > best_dev_accuracy:
                    best_dev_accuracy = self.val_acc[epoch]
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement == self.patience:
                        print(f'Training didn\'t improve for {self.patience} epochs\n'
                              f'Stopped Training at epoch {epoch}\n'
                              f'Best epoch: {epoch - self.patience}')
                        break

            self.print_metrics(epoch, start)

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
                      f'Train k-accuracy: {self.train_acc_k[epoch]:.3f}\n'
                      f'Time elapsed (remaining): {timeSince(start, (epoch+1) / NUM_EPOCHS)}')

            else:
                self.epoch_before_early_stop = epoch
                print(f'Epoch #{epoch + 1}:\n'
                      f'Validation Loss: {self.val_loss[epoch]:.4f}\n'                      
                      f'Validation accuracy: {self.val_acc[epoch]:.3f}\n'
                      f'Validation k-accuracy: {self.val_acc_k[epoch]:.3f}\n')

    def early_stopping_check(self, curr_epoch):
        if curr_epoch < EARLY_STOP_N :
            return False
        for i in range(0, EARLY_STOP_N):
            if self.val_acc[curr_epoch - i] - self.val_acc[curr_epoch - i - 1] >= EARLY_STOP_ACC_VALUE:
                return False
        print("made early stopping after epoch: ", curr_epoch)
        return True

    def evaluate(self, dataloader, val=False):
        total = 0.0
        correct = 0.0
        correct_k = 0.0
        epoch_val_loss = 0.0
        self.net.eval()

        with torch.no_grad():

            for inputs, labels in tqdm(dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.squeeze_()
                outputs = self.net(inputs)

                current_correct, current_total = calculate_accuracy(outputs, labels)
                current_k_correct, _ = top_k_accuracy(outputs, labels, k=self.k)

                correct_k += current_k_correct
                correct += current_correct
                total += current_total

                if val:
                    val_loss = CRITERION(outputs, labels.long())
                    epoch_val_loss += val_loss

        accuracy = correct / total
        k_accuracy = correct_k / total
        epoch_val_avg_loss = epoch_val_loss/len(self.train_dataloader)

        return accuracy, k_accuracy, epoch_val_avg_loss
