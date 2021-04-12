# import pandas as pd
from tqdm import tqdm
from handy_function import print_current_time
import torch
import time
import copy
from handy_function import timeSince,save_model, calculate_accuracy, top_k_accuracy


class TrainNet:
    def __init__(self, train_dataloader, net, optimizer, device, args, val_dataloader=None, save= False,  tr_bert_classifer = False,use_validation= True, k=5 ):

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.net = net
        #self.best_net = None
        self.device = device
        self.save = save
        self.use_validation = use_validation
        self.num_epochs = args.num_epochs
        self.criterion = args.criterion
        self.early_stop_n = args.early_stop_n
        self.early_stop_acc_value = args.early_stop_acc_value
        self.tr_bert_classifer = tr_bert_classifer
        self.k = k

        self.epoch_before_early_stop = 0
        self.val_best_acc_epoch = 0
        self.val_acc_value_before_eraly_stop = 0.0
        self.val_loss_value_before_eraly_stop = 0.0
        self.tr_bert_classifer = tr_bert_classifer
        self.val_best_acc_value = 0.0
        self.val_best_loss_value = 100000.0




        self.train_loss = [None] *  self.num_epochs
        self.val_loss = [None] *  self.num_epochs

        self.train_acc = [None] *  self.num_epochs
        self.train_acc_k = [None] * self.num_epochs

        self.val_acc = [None] *  self.num_epochs
        self.val_acc_k = [None] * self.num_epochs




        self.train_net(tr_bert_classifer)

    def train_net(self, tr_bert_classifer= False):

        start = time.time()
        print_current_time("starting to train classifier net")
        epoch_total_train_loss = 0.0  # Reset every epoch

        for epoch in range( self.num_epochs):

            for tr_batch in tqdm(self.train_dataloader):

                self.net.train()
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward
                loss, y_pred = self.forwad(tr_batch)

                # backward + optimize
                y_train = self.get_labels(tr_batch)
                loss = self.get_loss(loss, y_pred, y_train)
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
                self.update_best_val_loss_acc(self.val_acc[epoch], self.val_loss[epoch], epoch )
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
        if epoch % 5 == 0 or epoch == self.num_epochs-1:
            print()
            print("******************************")

            if train:
                print(f'Epoch #{epoch + 1}:\n'
                      f'Train Loss: {self.train_loss[epoch]:.4f}\n'
                      f'Train accuracy: {self.train_acc[epoch]:.4f}\n'
                      f'Train k-accuracy: {self.train_acc_k[epoch]:.3f}\n'
                      f'Time elapsed (remaining): {timeSince(start, (epoch + 1) / self.num_epochs)}')

            else:

                print(f'Epoch #{epoch + 1}:\n'
                      f'Validation Loss: {self.val_loss[epoch]:.4f}\n'
                      f'Validation accuracy: {self.val_acc[epoch]:.4f}\n' 
                      f'Validation k-accuracy: {self.val_acc_k[epoch]:.3f}\n')

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
        correct_k = 0.0
        self.net.eval()

        with torch.no_grad():

            for val_batch in tqdm(dataloader):
                loss,outputs = self.forwad(val_batch)

                labels = self.get_labels(val_batch)
                current_correct, current_total = calculate_accuracy(outputs, labels)
                current_k_correct, _ = top_k_accuracy(outputs, labels, k=self.k)

                correct_k += current_k_correct
                correct += current_correct
                total += current_total

                if val:
                    val_loss = self.criterion(outputs, labels.long())
                    epoch_val_loss += val_loss.item()

        accuracy = correct / total
        k_accuracy = correct_k / total
        epoch_val_avg_loss = epoch_val_loss/len(self.train_dataloader)

        return accuracy, k_accuracy, epoch_val_avg_loss

    def forwad(self, batch):

        loss = -1

        if self.tr_bert_classifer:

            b_input_ids  =  batch[0].to(self.device).long()
            b_input_mask =  batch[1].to(self.device)
            b_labels     =  batch[2].to(self.device).long()
            #b_labels = b_labels.squeeze_()

            loss, y_pred = self.net(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels,
                                 return_dict=False
                                    )
        else:
            x_train = batch[0]
            x_train = x_train.to(self.device)

            y_pred = self.net(x_train)


        return loss, y_pred



    def get_labels(self, batch):

        if self.tr_bert_classifer:
            labels = batch[2].to(self.device)

        else:
            labels = batch[1].to(self.device)

        labels = labels.squeeze_()

        return labels

    def get_loss(self, loss, y_pred, y_train):

        if self.tr_bert_classifer:
            return loss

        else:
            loss = self.criterion(y_pred, y_train.long())
            return loss

    def update_best_val_loss_acc(self,  last_epoch_acc_value,  last_epoch_loss_value, epoch):

        if last_epoch_acc_value > self.val_best_acc_value:
            self.val_best_acc_value= last_epoch_acc_value
            self.val_best_acc_epoch = epoch
            #self.best_net = copy.deepcopy(self.net)

        if last_epoch_loss_value < self.val_best_loss_value:
            self.val_best_loss_value =  last_epoch_loss_value


    # def predit_test(self, test_dataloader):
    #
    #     print_current_time("start to predict test labels")
    #
    #     total = 0.0
    #     correct = 0.0
    #
    #     with torch.no_grad():
    #
    #             for test_batch in tqdm(test_dataloader):
    #                 loss, outputs = self.forwad(test_batch)
    #
    #                 labels = self.get_labels(test_batch)
    #                 current_correct, current_total = calculate_accuracy(outputs, labels)
    #                 correct += current_correct
    #                 total += current_total
    #
    #     print_current_time("finsih to predict test labels")
    #
    #     accuracy = round(correct / total, 4)
    #
    #     print("the test accuracy value is: ", accuracy)
    #









