from handy_function import  print_current_time
from classification_net import ClassificationNet
from torch import optim
from train_net import TrainNet
import functools
import pandas as pd



class TrainedClassificationNetsParams:
    def __init__(self):

        self.trained_nets_params = []


    def train_net(self, device, args, embedding_dataloaders, index):

            classification_net = ClassificationNet(args).to(device)

            if args.weight_decay > 0:
                adam_optimizer = optim.Adam(classification_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            else:
                adam_optimizer = optim.Adam(classification_net.parameters(), lr=args.lr)

            training_net = TrainNet(train_dataloader=embedding_dataloaders.tr_dataloader,
                                                       optimizer=adam_optimizer,
                                                       device=device, net=classification_net,
                                                       val_dataloader=embedding_dataloaders.val_dataloader,
                                                      args= args

                                                       )

            self.save_params(index, args,  training_net.epoch_before_early_stop,  training_net.val_acc_value_before_eraly_stop,
                             training_net.val_loss_value_before_eraly_stop,
                             training_net.val_best_acc_value,   training_net.val_best_loss_value)


      
    def fitness(self,item):
       key = list(item)[0]
       return item.get(key, {}).get('val_acc_value_before_eraly_stop')



    def compare(self,item1, item2):
         return self.fitness(item1) - self.fitness(item2)



    def sort_params_by_val_acc(self):
        sorted(self.trained_nets_params, key=functools.cmp_to_key(self.compare))



    def save_params(self, index, args, epoch_before_early_stop,  val_acc_value_before_eraly_stop,
                    val_loss_value_before_eraly_stop, val_best_acc_value,  val_best_loss_value ):

        net_params_dict = {}
        net_params_dict[index] = {}
        net_params_dict[index]['p1'] = args.p1
        net_params_dict[index]['p2'] = args.p2
        net_params_dict[index]['fc1_output_size'] = args.fc1_output_size
        net_params_dict[index]['fc2_output_size'] = args.fc2_output_size
        net_params_dict[index]['batch_size'] = args.tr_batch_size
        net_params_dict[index]['lr'] = args.lr
        net_params_dict[index]['weight_decay'] = args.weight_decay

        net_params_dict[index]['epoch_before_early_stop'] = epoch_before_early_stop
        net_params_dict[index]['val_acc_value_before_eraly_stop'] =  val_acc_value_before_eraly_stop
        net_params_dict[index]['val_loss_value_before_eraly_stop'] = val_loss_value_before_eraly_stop
        net_params_dict[index]['val_best_acc_value'] =  val_best_acc_value
        net_params_dict[index]['val_best_loss_value'] = val_best_loss_value


        self.trained_nets_params.append(net_params_dict)



    def parms_to_csv(self):

        self.sort_params_by_val_acc()

        data = self.trained_nets_params

        final_df = pd.DataFrame()

        for id in range(0, len(data)):
            df = pd.DataFrame.from_dict(data[id])
            final_df = pd.concat([final_df, df], axis=1)

        print(final_df)

        final_df.to_excel('data.xlsx')


















