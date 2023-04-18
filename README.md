# Depth map estimation from stereo images

**Odin Hoff Gardå, April 2023**

## Introduction

The following repo is built on my [previous implementation of a group equivariant CNN for stereo images](https://github.com/odinhg/Group-Equivariant-Convolutional-Neural-Network-INF367A). Instead of binary classification, we are now trying to use the GCNN model to predict depth maps. The dataset used is part of the New Tsukuba Stereo Dataset created at [Tsukuba University's CVLAB](http://cvlab.cs.tsukuba.ac.jp). This dataset contains stereo images and depth maps (ground truth) taken from a rendered 3D scene and consists of 1800 images per camera in PNG format. Here, we only use the images rendered under fluorescent illumination.


