import time
import torch
import math
import os
from matplotlib import pyplot as plt
import numpy as np
from args import FIGURES_DIR, PARAMETERS_DIR
from datetime import datetime


def print_current_time(output=''):
    import datetime
    import pytz
    current_time = datetime.datetime.now(pytz.timezone('Israel'))
    if output == '':
        print("The current time is: ")
    else:
        print(output)
    print(current_time)


def move_x_and_y_cpu( x, y):
        x = x.cpu()
        y = y.cpu()
        return (x,y)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (%s)' % (asMinutes(s), asMinutes(rs))


def calculate_accuracy(outputs, labels):
    pred = torch.argmax(outputs, dim=1)
    total = labels.size(0)
    correct = (pred == labels).sum().item()
    return correct, total


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def save_model(model, epoch, postfix=''):
    torch.save(model, os.path.join(PARAMETERS_DIR, f'{epoch}{postfix}.pt'))


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


def plot_loss(loss, model_name):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.figure(figsize=(12, 8))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.tight_layout()
    plt.plot(loss)
    plt.title(f'{model_name} training loss')
    plt.xticks(range(len(loss)), range(1, len(loss) + 1))
    # plt.yticks(np.around(np.linspace(0.0, 1.0, num=11), decimals=1))
    # plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, f'{model_name}_loss_{current_time}.png'))
    plt.show()
