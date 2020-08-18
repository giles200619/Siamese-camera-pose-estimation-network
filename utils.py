# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 22:04:04 2020

@author: giles
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import random
import shutil
import open3d as o3d


def get_object_T_camera(x: float, y: float, z: float) -> np.ndarray:
    z_vector = np.array([-x, -y, -z])
    e_z = z_vector / np.linalg.norm(z_vector)
    x_vector = np.cross(e_z, np.array([0,0,1]))
    e_x = x_vector / np.linalg.norm(x_vector)
    e_y = np.cross(e_z, e_x)

    camera_position = np.array([x,y,z])

    object_T_camera = np.c_[e_x, e_y, e_z, camera_position]
    return object_T_camera

def spherical_to_cartesian(azimuth: float, elevation: float, distance: float = 1.0):
    #if azimuth > 2 * np.pi or elevation > 2 * np.pi:
        #warnings.warn('Expects radians, received {} for azimuth and {} for elevation'.format(azimuth, elevation))
    z = distance * np.sin(elevation)

    d_cos = distance * np.cos(elevation)
    x = d_cos * np.cos(azimuth)
    y = d_cos * np.sin(azimuth)

    return x, y, z

def pose_from_filename(filename: str) -> np.ndarray:
    azimuth_degree, elevation_degree = tuple(float(v) for v in filename.split('.')[0].split('_')[-2:])
    azimuth_degree *= -10

    azimuth_rad, elevation_rad = np.deg2rad(azimuth_degree), np.deg2rad(elevation_degree)
    x, y, z = spherical_to_cartesian(azimuth_rad, elevation_rad)

    object_T_camera = get_object_T_camera(x, y, z)
    return object_T_camera
    
def RotationMatrix6D(img1,img2):
    R01 = pose_from_filename(img1)
    R02 = pose_from_filename(img2)
    R12 = np.transpose(R01[:,0:3]) @ R02[:,0:3]
    
    T12 = np.subtract(R02[:,3],R01[:,3])
    
    return np.hstack((np.reshape(R12[:,0:2],(1,6)),np.reshape(T12,(1,3))))

def Rfrom6D(tensor):
    b1 = np.array([tensor.cpu().numpy()[0],tensor.cpu().numpy()[2],tensor.cpu().numpy()[4]])
    a2 = np.array([tensor.cpu().numpy()[1],tensor.cpu().numpy()[3],tensor.cpu().numpy()[5]])
    b1 = b1/np.linalg.norm(b1)
    b2 = a2 - (b1 @ a2)*b1
    b2 = b2/np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    
    return np.c_[b1,b2,b3]

def elvaluateR(groundtruth, outputs):
    I = np.identity(3)
    error = 0
    for i in range(groundtruth.shape[0]):
        R_p = Rfrom6D(outputs[i].detach())
        R_t = Rfrom6D(groundtruth[i])
        error+=np.linalg.norm(R_p @ np.transpose(R_t) - I,'fro') 
    return error/groundtruth.shape[0]

def splitTrainTest(datadir):
    im_list = os.listdir(datadir)
    test_list = random.sample(im_list, int(len(im_list)*0.1))
    for im in im_list:
        if im not in test_list:
            shutil.move(os.path.join(datadir,im),'./dataset/train')
        else:
            shutil.move(os.path.join(datadir,im),'./dataset/test')
            

#visualize3D(filename_list[0][0],filename_list[0][1],outputs[0].detach())
#output = outputs[i].detach()
#black points represent the two iuput camera pose, red point represents the estimated camera2.
def visualize3D(img1,img2,output):
    sphere_points=[]
    for x in range(0,100):
        y = np.sqrt(10000-(x*x))
        sphere_points.append(np.array([x/100,y/100,0]))
        sphere_points.append(np.array([-x/100,y/100,0]))
        sphere_points.append(np.array([x/100,-y/100,0]))
        sphere_points.append(np.array([-x/100,-y/100,0]))
        
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])
    R01 = pose_from_filename(img1)
    R02 = pose_from_filename(img2)
    R12 = Rfrom6D(output)
    R12 = R01[:,0:3] @ R12
    R12 = np.hstack((R12,np.array([[output.numpy()[6]+R01[0][3]],[output.numpy()[7]+R01[1][3]],[output.numpy()[8]+R01[2][3]]])))
    R12 = np.vstack((R12,np.array([0,0,0,1])))
    
    camera1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])
    camera2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])
    camera_12 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])
    camera1 = camera1.transform(np.vstack((R01,np.array([0,0,0,1]))))
    camera2 = camera2.transform(np.vstack((R02,np.array([0,0,0,1]))))
    camera_12 = camera_12.transform(R12)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sphere_points)
    
    ori_c1 = o3d.geometry.PointCloud()
    ori_c1.points = o3d.utility.Vector3dVector([np.array([R01[0][3],R01[1][3],R01[2][3]+0.1])])
    ori_c1.colors = o3d.utility.Vector3dVector([np.array([0,0,0])])
    
    ori_c2 = o3d.geometry.PointCloud()
    ori_c2.points = o3d.utility.Vector3dVector([np.array([R02[0][3],R02[1][3],R02[2][3]+0.1])])
    ori_c2.colors = o3d.utility.Vector3dVector([np.array([0,0,0])])
    
    ori_pre = o3d.geometry.PointCloud()
    ori_pre.points = o3d.utility.Vector3dVector([np.array([R12[0][3],R12[1][3],R12[2][3]+0.1])])
    ori_pre.colors = o3d.utility.Vector3dVector([np.array([255,0,0])]) 
    
    o3d.visualization.draw_geometries([world_frame, camera1, camera2,camera_12,pcd,ori_c1,ori_c2,ori_pre])
            
### test single image ###
def single_RotationMatrix6D(img1,img2):
    R01 = pose_from_filename(img1)
    T01 = R01[:,3]
    
    return np.hstack((np.reshape(R01[:,0:2],(1,6),'F'),np.reshape(T01,(1,3))))

def singleimageR(filename_list):
    gt = np.zeros((len(filename_list),9))
    for i in range(len(filename_list)):
        gt[i,:] = single_RotationMatrix6D(filename_list[i][0],filename_list[i][1])
        
    return torch.from_numpy(gt)
###