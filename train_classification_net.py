# import pandas as pd
from tqdm import tqdm
from handy_function import print_current_time
import torch
import time
from handy_function import timeSince, calculate_accuracy, save_model
from args import CRITERION, EARLY_STOP_N, EARLY_STOP_LOSS_VALUE, NUM_EPOCHS, VALIDATION_RATIO


class TrainClassificationNet:
    def __init__(self, train_dataloader, net, optimizer, device, val_dataloader=None, save=True):

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.net = net.to(device)
        self.device = device
        self.save = save
        self.use_validation = VALIDATION_RATIO > 0

        self.train_loss = [None] * NUM_EPOCHS
        self.val_loss = [None] * NUM_EPOCHS

        self.train_acc = [None] * NUM_EPOCHS
        self.val_acc = [None] * NUM_EPOCHS

        self.train_net()

    def train_net(self):
        start = time.time()
        print_current_time("starting to train classifier net")
        epoch_total_train_loss = 0.0  # Reset every epoch

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

            # loss
            self.train_loss[epoch] = epoch_total_train_loss / len(self.train_dataloader)
            epoch_total_train_loss = 0.0

            # accuracy
            epoch_total_train_acc = self.evaluate(self.train_dataloader)
            self.train_acc[epoch] = epoch_total_train_acc
            test_accuracy = self.evaluate(self.val_dataloader)
            self.val_acc[epoch] = test_accuracy

            self.print_metrics(epoch, start)
            # print(f'Epoch #{epoch}:\n'
            #       f'Last batch Loss: {loss.item():.4f}\n'
            #       f'Train accuracy: {epoch_total_train_acc:.3f}\n'
            #       f'Test accuracy: {test_accuracy:.3f}\n'
            #       f'Time elapsed (remaining): {timeSince(start, (epoch+1) / NUM_EPOCHS)}')

            if self.save:
                save_model(self.net, epoch)

            # if self.use_validation:
            #     self.check_validation(epoch)
            #
            #     if self.early_stopping_check(epoch):
            #         break

    def print_metrics(self, epoch, start, train=True):
        if epoch % 20 == 0:
            print()
            print("******************************")

            if train:
                print("accuracy value: ", self.train_acc[epoch])
                print(f'Epoch #{epoch+1}:\n'
                      f'Train Loss: {self.train_loss[epoch]:.4f}\n'
                      f'Train accuracy: {self.train_acc[epoch]:.3f}\n'
                      f'Validation accuracy: {self.val_acc[epoch]:.3f}\n'
                      f'Time elapsed (remaining): {timeSince(start, (epoch+1) / NUM_EPOCHS)}')

            else:
                self.epoch_before_eraly_stop = epoch
                print_current_time("Validation metrics for epoch " + str(epoch) + ":\n")
                print("Loss value: ", self.val_loss[epoch])
                print("accuracy value: ", self.val_acc[epoch])

    def early_stopping_check(self, curr_epoch):
        if curr_epoch < EARLY_STOP_N - 1:
            return False

        loss_improved = False

        for i in range(0, EARLY_STOP_N):
            if self.val_loss[curr_epoch - i] - self.val_loss[curr_epoch - i - 1] <= EARLY_STOP_LOSS_VALUE:
                loss_improved = True
                return False

        if not loss_improved:
            print("made early stopping after epoch: ", curr_epoch)
            return True

    def evaluate(self, dataloader):
        total = 0
        correct = 0
        self.net.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.squeeze_()
                outputs = self.net(inputs)
                current_correct, current_total = calculate_accuracy(outputs, labels)
                correct += current_correct
                total += current_total

        accuracy = correct / total
        return accuracy











