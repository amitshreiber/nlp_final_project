import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationNet(torch.nn.Module):

    def __init__(self, args, input_size):
        super(ClassificationNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, args.fc1_output_size)
        self.BN1 = torch.nn.BatchNorm1d( args.fc1_output_size)
        self.p1 = args.p1
        self.dropout_1 = nn.Dropout(args.p1)

        self.fc2 = torch.nn.Linear( args.fc1_output_size, args.fc2_output_size)
        self.BN2 = torch.nn.BatchNorm1d( args.fc2_output_size)
        self.p2 = args.p2
        self.dropout_2 = nn.Dropout(args.p2)

        self.fc3 = torch.nn.Linear( args.fc2_output_size, args.class_number)

    def forward(self, x):

        x = self.fc1(x)
        x = self.BN1(x)
        x = F.relu(x)

        if self.p1 > 0:
            x = self.dropout_1(x)

        x = self.fc2(x)
        x = self.BN2(x)
        x = F.relu(x)

        if self.p2 > 0:
            x = self.dropout_2(x)

        y_pred = self.fc3(x)
        return y_pred
