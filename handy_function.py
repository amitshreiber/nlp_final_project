import time
import torch
import math
import os
import numpy as np
from directories import FIGURES_DIR, PARAMETERS_DIR



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


