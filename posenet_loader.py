# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 22:53:56 2020

@author: giles
"""
from __future__ import print_function, division
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import RotationMatrix6D
import random

class ImagePairDataset(Dataset):
    def __init__(self, dir_path):
        self.name_pair = self.dataNamePair(dir_path)
        random.shuffle(self.name_pair)
        self.length = len(self.name_pair)
        self.path = dir_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = {}
        
        filename_list = self.name_pair[index]
        input1, input2 = self.getImageTensor(filename_list)
        groundtruth = self.groundTruthTensor(filename_list)

        
        data = {'input1': input1,
                'input2': input2,
                'groundtruth':groundtruth
                }
        
        return data
    
    def dataNamePair(self,datadir):
        im_list = os.listdir(datadir)
        name_list = [] 
        index_list = [] 
        current_list=[] 
        for i in range(len(im_list)):
            name = im_list[i]
            split = name.split("_")
            if split[0] not in name_list:
                index_list.append(current_list)
                current_list=[]
                name_list.append(split[0])
            current_list.append(i)
            if i == len(im_list)-1:
                index_list.append(current_list)
        index_list.pop(0)
        
        name_pair = []
        for cur_list in index_list:
            length = len(cur_list)
            for j in range(length-1):   #index1 = cur_list[j]
                for k in range(length-1-j):    #index2 = cur_list[k]  
                    name1 = im_list[cur_list[j]]
                    name2 = im_list[cur_list[k+j+1]]
                    name_pair.append([name1,name2])
        
        return name_pair

    def getImageTensor(self,filename_list):
    
        trans = transforms.Compose([
    	transforms.Resize((100,100)), 
    	transforms.ToTensor(), 
    	])
        
        img1 = Image.open(os.path.join(self.path, filename_list[0]))
        img2 = Image.open(os.path.join(self.path, filename_list[1]))
        img1 = trans(img1)
        img2 = trans(img2)
            
        return img1, img2

    def groundTruthTensor(self,filename_list):
        gt = RotationMatrix6D(filename_list[0],filename_list[1])
        
        return torch.from_numpy(gt)
    