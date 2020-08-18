# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 01:42:10 2020

@author: giles
Siamese Pose Net
"""
from __future__ import print_function
import torch
import torch.nn as nn
import os
import numpy as np
import argparse
import open3d as o3d
from posenet_loader import ImagePairDataset
from model import SiamesePoseNet
from utils import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--train', dest='train',  default=1 ,type=int, help='1 for train, 2 for continue training, 0 for test')
parser.add_argument('--batch_size', dest='batch_size',  default=10 ,type=int, help='batch size')
parser.add_argument('--epoch', dest='epoch',  default=40 ,type=int, help='epoch')
parser.add_argument('--train_dir', dest='train_dir',  default='./dataset/chairs/train' , help='training data dir')
parser.add_argument('--test_dir', dest='test_dir',  default='./dataset/chairs/test' , help='testing data dir')
parser.add_argument('--save_frequency', dest='save_frequency',  default=10000,type=int, help='save model every # of iteration')
parser.add_argument('--print_frequency', dest='print_frequency',  default=100,type=int, help='print loss every # of iteration')
parser.add_argument('--checkpoint', dest='checkpoint',  default='./checkpoint/0_10000.pth' , help='checkpoint path')
args = parser.parse_args()

if __name__ == "__main__":    
    print("cuda avail:",torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if not os.path.isdir('./checkpoint'):
        os.mkdir('./checkpoint')
    
    net = SiamesePoseNet()
    net = net.float()
    net.to(device)
    #pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma=0.3)
    
    #train
    if args.train == 1:
        #name_pair = dataNamePair(args.train_dir)
        #random.shuffle(name_pair)
        #iterations = int(len(name_pair)/args.batch_size)
        
        data = ImagePairDataset(args.train_dir)
        dataloader = torch.utils.data.DataLoader(data,batch_size=args.batch_size,
                        shuffle=True,num_workers=2)
        epoch=0
        iterations = data.length
        print("Training iterations:",iterations)
        print("Epoch:", args.epoch)
        print("Start Training...")
        losses = []
        
    if args.train == 2:
        data = ImagePairDataset(args.train_dir)
        dataloader = torch.utils.data.DataLoader(data,batch_size=args.batch_size,
                        shuffle=True,num_workers=2)
        iterations = data.length
        print("Continue Training...")
        
        if not (os.path.isfile(args.checkpoint) or os.path.splitext(args.checkpoint)[1] == '.pth'):
            raise OSError('Invalid checkpoint.')
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        losses = checkpoint['losses']
        print("Training iterations:",iterations)
        print("Epoch:", epoch,'/',args.epoch)
        
    if args.train != 0:
        for e in range(epoch,args.epoch):
            for i, sample_batched in enumerate(dataloader):
                
                input1 = sample_batched['input1']
                input2 = sample_batched['input2']
                groundtruth = sample_batched['groundtruth']
                
                input1 = input1.to(device)
                input2 = input2.to(device)
                groundtruth = groundtruth.to(device)
                groundtruth = torch.squeeze(groundtruth, 1)
                l1 = nn.L1Loss()
                optimizer.zero_grad()
                outputs = net.forward(input1.float(), input2.float())
                loss = l1(outputs, groundtruth)
                loss.backward()
                optimizer.step()
                
                
                if (e*iterations+i)%args.print_frequency == args.print_frequency-1:
                    print(loss.item(),'iter:',i+1,f'[{e}/{args.epoch}]')
                    losses.append(loss.item())
                                
                if (e*iterations+i)%args.save_frequency == args.save_frequency-1:
                    torch.save({
                                'epoch': e,
                                'model_state_dict': net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss,
                                'iteration': i,
                                'losses':losses,
                                }, f'./checkpoint/{e}_{i+1}.pth')
            torch.save({'epoch': e+1,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'iteration': 0,
                        'losses':losses,
                        }, f'./checkpoint/{e}_{i+1}.pth')        
        print('Finished training.')
     
    #test
    if args.train == 0:
        print("start testing...")        
        if not (os.path.isfile(args.checkpoint) or os.path.splitext(args.checkpoint)[1] == '.pth'):
            raise OSError('Invalid checkpoint.')
        data = ImagePairDataset(args.test_dir)
        dataloader = torch.utils.data.DataLoader(data,batch_size=args.batch_size,
                        shuffle=False,num_workers=2)
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['model_state_dict'])
        
        iterations = data.length
        print("number to evaluate:", iterations)
        error = 0
        for i, sample_batched in enumerate(dataloader):
            input1 = sample_batched['input1']
            input2 = sample_batched['input2']
            groundtruth = sample_batched['groundtruth']
            groundtruth = torch.squeeze(groundtruth, 1)
            input1 = input1.to(device)
            input2 = input2.to(device)
            groundtruth = groundtruth.to(device)
            
            outputs = net.forward(input1.float(), input2.float())
            error += elvaluateR(groundtruth,outputs)
            if(i%100==0):
                print("processing...",i*args.batch_size)
        print("mean_error:", error/iterations)

        #show result using open3d:
        #visualize3D(filename_list[0][0],filename_list[0][1],outputs[0].detach())
