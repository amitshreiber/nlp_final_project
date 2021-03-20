import torch
import torch.nn as nn
import torch.nn.functional as F
from args import *


class args_instance:


    def __init__(self,lr,  p1, p2, fc1_output_size, fc2_output_size,
                 tr_batch_size, val_batch_size, weight_decay):
        # NN architecture args

        self.input_size =  INPUT_SIZE
        self.class_number = CLASS_NUMBER
        self.p1 = p1
        self.p2 = p2
        self.fc1_output_size = fc1_output_size
        self.fc2_output_size = fc2_output_size

        # Train-test split args
        self.validation_ratio  = VALIDATION_RATIO
        self.test_ratio =  TEST_RATIO

        # NN training args
        self.tr_batch_size=   tr_batch_size
        self.val_batch_size=  val_batch_size
        self.num_epochs =     NUM_EPOCHS

        # NN back propagation args
        self.lr = lr
        self.weight_decay  = weight_decay
        self.criterion =  CRITERION

        # NN early stopping args
        self.early_stop_n = EARLY_STOP_N
        self.early_stop_acc_value = EARLY_STOP_ACC_VALUE


class ArgsComb:
    def __init__(self):
     n =  len(P1) * len(P2) * len(FC1_OUTPUT_SIZE) * len(FC2_OUTPUT_SIZE) * len(TR_BATCH_SIZE )  * len(WEIGHT_DECAY ) * len(LR)
     self.args_combs_list = [None] * n

     i = 0
     for p1 in P1:
         for p2 in P2:
             for  fc1_output_size in FC1_OUTPUT_SIZE:
                 for fc2_output_size in FC2_OUTPUT_SIZE:
                     for tr_batch_size in TR_BATCH_SIZE:
                         val_batch_size = tr_batch_size
                         for weight_decay in WEIGHT_DECAY:
                             for lr in LR:
                                 args=  args_instance(p1 = p1, p2= p2, fc1_output_size = fc1_output_size,
                                                                fc2_output_size = fc2_output_size, tr_batch_size= tr_batch_size,
                                                                val_batch_size= val_batch_size, weight_decay = weight_decay, lr = lr)

                                 self.args_combs_list [i] = args
                                 i += 1
                                 print(i)






