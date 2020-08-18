# Siamese-camera-pose-estimation-network
A Pytorch implementation of a supervised Siamese structure network for camera pose rotation estimation. Given two rgb images, this network estimates the transformation matrix from the first camera pose to the second. The Rotation matrix is represented in 6D to ensure continuity.[1] 

![Architecture](/example/architecture.PNG)
![visual_cpe](/example/visual_cpe.PNG)

## Dependencies
* pytorch 1.5.1 
* torchvision 0.6.1 
* open3d 0.9.0.0

## Data
The dataloader is based on the Shapenet dataset naming convention: {model name}\_{azimuth/10}\_{elevation}.png

## Reference
[1] Zhou, Yi, et al. "On the continuity of rotation representations in neural networks." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
