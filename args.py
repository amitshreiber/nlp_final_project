import torch.nn.functional
import os

"""
================
constants
================
"""

# NN architecture args
#HIDDEN_DIM = 50
INPUT_SIZE = 768
CLASS_NUMBER = 19
p= 0.5


# Train-test split args
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2

# NN training args
TR_BATCH_SIZE = 256
VAL_BATCH_SIZE = 256
NUM_EPOCHS = 10000000

# NN back propagation args
LR = 0.01
WEIGHT_DECAY = 5e-4
CRITERION = torch.nn.CrossEntropyLoss()

# NN early stopping args
EARLY_STOP_N = 5
EARLY_STOP_LOSS_VALUE = 0



# directories
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))  # This is your Project Root
PARAMETERS_DIR = os.path.join(ROOT_DIR, 'parameters')
FIGURES_DIR = os.path.join(ROOT_DIR, 'figures')

