from handy_function import  print_current_time
from classification_net import ClassificationNet
from torch import optim
from train_classification_net import TrainClassificationNet
import functools



class TrainedClassificationNets:
    def __init__(self, device):

        self.trained_nets = []






    def train_net(self, device, args, embedding_dataloaders):

            classification_net = ClassificationNet(args).to(device)
            self.args = args
            if args.weight_decay > 0:
                adam_optimizer = optim.Adam(classification_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            else:
                adam_optimizer = optim.Adam(classification_net.parameters(), lr=args.lr)

            training_net = TrainClassificationNet(train_dataloader=embedding_dataloaders.tr_dataloader,
                                                       optimizer=adam_optimizer,
                                                       device=device, net=classification_net,
                                                       val_dataloader=embedding_dataloaders.val_dataloader,

                                                       )
            self.trained_nets.append(training_net)

     

      
    def fitness(self,item):
           return item.val_loss_value_before_eraly_stop

    def compare(self,item1, item2):
         return self.fitness(item1) - self.fitness(item2)


    def sort_net_by_val_acc(self):
        sorted(self.trained_nets, key=functools.cmp_to_key(self.compare))


