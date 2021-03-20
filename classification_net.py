import torch
import torch.nn as nn
import torch.nn.functional as F
from args import CLASS_NUMBER, INPUT_SIZE, p1, p2


class ClassificationNet(torch.nn.Module):
    def __init__(self):
        super(ClassificationNet, self).__init__()
        self.fc1 = torch.nn.Linear(INPUT_SIZE, 256)
        self.BN1 = torch.nn.BatchNorm1d(256)
        self.dropout_1 = nn.Dropout(p1)

        self.fc2 = torch.nn.Linear(256,64)
        self.BN2 = torch.nn.BatchNorm1d(64)
        self.dropout_2 = nn.Dropout(p2)


        self.fc3 = torch.nn.Linear(64, CLASS_NUMBER)




    def forward(self, x):

        x = self.fc1(x)
        x = self.BN1(x)
        x = F.relu(x)
        x= self.dropout_1(x)

        x = self.fc2(x)
        x = self.BN2(x)
        x = F.relu(x)
        x = self.dropout_2(x)

        y_pred = self.fc3(x)
        return y_pred
