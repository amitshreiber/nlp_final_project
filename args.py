import torch.nn.functional
import os
import numpy as np


def get_learning_rate():

    learning_rate1 = np.arange(0.0001, 0.0021, 0.0001).tolist()
    learning_rate2 = np.arange(0.003, 0.021, 0.001).tolist()
    learning_rate3 = np.arange(0.03, 0.11, 0.01).tolist()

    return learning_rate1 + learning_rate2 + learning_rate3



"""
================
constants
================
"""

# NN architecture args
INPUT_SIZE = 768
CLASS_NUMBER = 19
P1= [0,0.3, 0.4 , 0.5]
P2 = [0,0.25, 0.1, 0.15, 0.2]
FC1_OUTPUT_SIZE = [512, 256, 128]
FC2_OUTPUT_SIZE = [64, 32, 16]


# Train-test split args
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2

# NN training args
TR_BATCH_SIZE =  [32,64,128,256,512]
VAL_BATCH_SIZE = [32,64,128,256,512]
NUM_EPOCHS = 10000000

# NN back propagation args
LR =  get_learning_rate()
WEIGHT_DECAY = [0.005, 0]
CRITERION = torch.nn.CrossEntropyLoss()

# NN early stopping args
EARLY_STOP_N = 3
EARLY_STOP_ACC_VALUE = 0



# directories
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))  # This is your Project Root
PARAMETERS_DIR = os.path.join(ROOT_DIR, 'parameters')
FIGURES_DIR = os.path.join(ROOT_DIR, 'figures')








