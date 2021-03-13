import torch
import torch.nn.functional as F


class classification_net(torch.nn.Module):

   def __init__(self, args):
      super(classification_net, self).__init__()
      self.fc1 = torch.nn.Linear(768,512)
      self.fc2 = torch.nn.Linear(512, 256)
      self.fc3 =  torch.nn.Linear(256,128)
      self.fc4 = torch.nn.Linear(128, 64)
      self.fc5 = torch.nn.Linear(64, 32)
      self.fc6 =  torch.nn.Linear(32, args.class_number)


      #self.dropout = nn.Dropout(0.5)

   def forward (self,x):

     x = F.relu(self.fc1(x))
     x = F.relu(self.fc2(x))
     x = F.relu(self.fc3(x))
     x = F.relu(self.fc4(x))
     x = F.relu(self.fc5(x))
     y_pred = (self.fc6(x))
     return y_pred