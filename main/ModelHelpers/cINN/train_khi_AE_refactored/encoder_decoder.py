import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

#simple encodr design
class Encoder(nn.Module):
    def __init__(self, zdim, input_dim=3, use_deterministic_encoder=False):
        super(Encoder, self).__init__()
        self.use_deterministic_encoder = use_deterministic_encoder
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        if self.use_deterministic_encoder:
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc_bn1 = nn.BatchNorm1d(256)
            self.fc_bn2 = nn.BatchNorm1d(128)
            self.fc3 = nn.Linear(128, zdim)
        else:
            #basically, creates a seperate mapping 
            #for variance and mean for use_deterministic_encoder
            #only has one mapping which is mean.  
            # Mapping to [c], cmean
            self.fc1_m = nn.Linear(512, 256)
            self.fc2_m = nn.Linear(256, 128)
            self.fc3_m = nn.Linear(128, zdim)
            self.fc_bn1_m = nn.BatchNorm1d(256)
            self.fc_bn2_m = nn.BatchNorm1d(128)

            # Mapping to [c], cmean
            self.fc1_v = nn.Linear(512, 256)
            self.fc2_v = nn.Linear(256, 128)
            self.fc3_v = nn.Linear(128, zdim)
            self.fc_bn1_v = nn.BatchNorm1d(256)
            self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        #x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        if self.use_deterministic_encoder:
            ms = F.relu(self.fc_bn1(self.fc1(x)))
            ms = F.relu(self.fc_bn2(self.fc2(ms)))
            ms = self.fc3(ms)
            #mean and variance.
            m, v = ms, 0
        else:
            m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
            m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
            m = self.fc3_m(m)
            v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
            v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
            v = self.fc3_v(v)
            v = F.softplus(v) + 1e-8

        return m, v

#decoder design
class MLP_Decoder(nn.Module):
    def __init__(self,zdim,n_point,point_dim):
        super(MLP_Decoder,self).__init__()
        self.zdim = zdim
        self.n_point = n_point
        self.point_dim = point_dim
        self.n_point_3 = self.point_dim * self.n_point
        self.fc1 = nn.Linear(self.zdim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.final = nn.Linear(256,self.n_point_3)
    
    def forward(self,z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        output  =  self.final(x)
        output = output.reshape(-1,self.point_dim,self.n_point)
        return output
