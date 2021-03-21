from matplotlib import pyplot as plt
from datetime import datetime
import os
import numpy as np
from args import FIGURES_DIR


def plot_accuracies(train_accs, test_accs, model_name):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.figure(figsize=(12, 8))
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    # plt.tight_layout()
    plt.plot(train_accs, label='train')
    plt.plot(test_accs, label='validation')
    plt.title(f'{model_name} accuracy')
    plt.xticks(range(len(train_accs)), range(1, len(train_accs) + 1))
    plt.yticks(np.around(np.linspace(0.0, 1.0, num=11), decimals=1))
    plt.legend()
    # plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, f'{model_name}_accuracy_{current_time}.png'))
    plt.show()


def plot_loss(train_loss, val_loss, model_name):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.figure(figsize=(12, 8))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.tight_layout()
    plt.plot(train_loss,  label='training')
    plt.plot(val_loss,   label='validation')
    plt.title(f'{model_name} loss')
    plt.xticks(range(len(train_loss)), range(1, len(train_loss) + 1))
    plt.legend()
    # plt.yticks(np.around(np.linspace(0.0, 1.0, num=11), decimals=1))
    # plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, f'{model_name}_loss_{current_time}.png'))
    plt.show()
