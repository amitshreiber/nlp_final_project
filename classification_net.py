import torch
import torch.nn as nn
import torch.nn.functional as F
from args import CLASS_NUMBER, INPUT_SIZE, p


class ClassificationNet(torch.nn.Module):
    def __init__(self):
        super(ClassificationNet, self).__init__()
        self.fc1 = torch.nn.Linear(INPUT_SIZE, 256)
        self.fc2 = torch.nn.Linear(256,128)
        self.fc3 = torch.nn.Linear(128, 32)
        self.fc4 = torch.nn.Linear(32, CLASS_NUMBER)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x= self.dropout(x)
        x = F.relu(self.fc2(x))
        #x= self.dropout(x)
        x = F.relu(self.fc3(x))
        x= self.dropout(x)
        y_pred = (self.fc4(x))
        return y_pred
