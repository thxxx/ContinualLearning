import torch.nn as nn
import torch

### CNN model definition
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layer = nn.Sequential(
        nn.Linear(32*16*16, 2048),
        nn.ReLU(inplace=True),
        nn.Linear(2048, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 100),
        )
        self.fc_layers=[]
            
    def forward(self, x, conv=False):
        conv_x = self.conv(x)
        conv_x = conv_x.view(-1, 32*16*16)
        if conv:
            return conv_x
        x = self.fc_layer(conv_x)
        x = torch.log_softmax(x, dim=1)
        return x
    
    def continues(self, x):
        features = self.forward(x, conv=True)
        prob_x=0
        vals_list=[]
        idx_list = []
        diffs=[]
        for fc_layer in self.fc_layers:
            x = fc_layer(features)
            prob_x = torch.log_softmax(x, dim=1) # [32, 2]
            vals, predicted = torch.max(prob_x, 1)
            vals_list.append(vals)
            idx_list.append(predicted)
            diffs.append(abs(torch.subtract(prob_x[:, 0], prob_x[:, 1])))
        
        return vals_list, idx_list, diffs