# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 01:42:10 2020

@author: giles
Siamese Pose Net
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
 
class SiamesePoseNet(nn.Module):
    
    def __init__(self):
        super(SiamesePoseNet, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding=1) 
        self.conv2 = nn.Conv2d(32,64,3,stride=1)
        self.conv3 = nn.Conv2d(64,128,3,stride=1)
        self.conv4 = nn.Conv2d(128,256,3,stride=1)
        self.conv5 = nn.Conv2d(256,256,4)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        
        self.pool = nn.MaxPool2d(2,2)
        
        
        self.fc1 = nn.Linear(512,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,9)
        
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features        
    
    def forward(self, input1,input2):        
        out1 = self.pool(self.bn1(F.relu(self.conv1(input1))))
        out1 = self.pool(self.bn2(F.relu(self.conv2(out1))))
        out1 = self.pool(self.bn3(F.relu(self.conv3(out1))))
        out1 = self.pool(self.bn4(F.relu(self.conv4(out1))))
        out1 = F.relu(self.conv5(out1)) #1*1*256
        
        out2 = self.pool(self.bn1(F.relu(self.conv1(input2))))
        out2 = self.pool(self.bn2(F.relu(self.conv2(out2))))
        out2 = self.pool(self.bn3(F.relu(self.conv3(out2))))
        out2 = self.pool(self.bn4(F.relu(self.conv4(out2))))
        out2 = F.relu(self.conv5(out2)) #1*1*256
        
        out = torch.cat((out1,out2),1)
        out = out.view(out.shape[0],-1)
        
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out) ##
        
        return out
