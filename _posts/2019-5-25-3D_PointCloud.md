---
layout:     post
title:      3D点云
subtitle:   点云初步认识
date:       2019-05-19
author:     Kai
header-img: img/home-bg-o.jpg
catalog: true
tags:
    - 点云
---

## 前言

点云的一些基础知识

## 激光雷达类型
* 机械式雷达
* MEMS式雷达

# 3D点云数据来源
* CAD模型
* 雷达传感器
* RGBD相机

## 点云数据特点
* 简单，有 x,y,z,i (三维坐标 加强度)
* 稀疏，约是图片数据的7%
* 无序
* 精度高，误差在正负2cm

## 图像和点云的比较

* 点云：简单，精确，适合几何感知
* 图像：有丰富的语义感知信息
* 坐标系不一样 ![坐标系](http://ronny.rest/media/tutorials/lidar/point_cloud_coordinates/photo_vs_lidar_axes.png)

## 点云数据集
* [KITTI](http://www.cvlibs.net/datasets/kitti/)
* [阿波罗](http://apolloscape.auto/) 
* [cityscapes](https://www.cityscapes-dataset.com/)
* [berkeley](https://bdd-data.berkeley.edu/)

## KITTI数据集

### KITTI 数据急简介
KITTI数据集由德国卡尔斯鲁厄理工学院和丰田美国技术研究院联合创办，是目前国际上最大的自动驾驶场景下的计算机视觉算法评测数据集。该数据集用于评测立体图像(stereo)，光流(optical flow)，视觉测距(visual odometry)，3D物体检测(object detection)和3D跟踪(tracking)等计算机视觉技术在车载环境下的性能。KITTI包含市区、乡村和高速公路等场景采集的真实图像数据，每张图像中最多达15辆车和30个行人，还有各种程度的遮挡与截断。整个数据集由389对立体图像和光流图，39.2 km视觉测距序列以及超过200k 3D标注物体的图像组成，以10Hz的频率采样及同步。

### 数据采集平台
KITTI数据采集平台包括2个灰度摄像机，2个彩色摄像机，一个Velodyne 3D激光雷达，4个光学镜头，以及1个GPS导航系统

![kitti数据采集平台](https://img-blog.csdn.net/20180521105216125?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3N5eXlhbw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

### KITTI数据下载
由于被墙没法下载，从网上找的资源，放在自己的百度云上了，[地址](https://pan.baidu.com/s/1Cekq77X_wtIElHKxC0FlsA)


## 引用
[Processing Point Clouds](http://ronny.rest/tutorials/module/pointclouds_01)








