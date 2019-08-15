---
layout:     post
title:      检测-Bounding Box Regression
subtitle:   边框回归的理解
date:       2019-08-15
author:     Kai
header-img: img/home-bg-o.jpg
catalog: true
tags:
    - Bounding Box Regression 
---

## 前言
一直对检测的边框回归不是很理解。自从看了这篇[博客](https://blog.csdn.net/zijin0802034/article/details/77685438)，对Bounding Box Regression 有了一定的理解。这篇博客记录一下对它的理解。按照参照的博客也先列出几个问题：

- 为什么要边框回归？
- 什么是边框回归？
- 边框回归怎么做的？

## 为什么要边框回归
<img src="{{ site.baseurl }}/img/2019-8-15-detection-Bounding Box Regression/Bounding-box-Regression.png" /> 

蓝色框是真值，红色框是Region Proposal。虽然找到了飞机，但是IOU<0.5 ，非常可以，这相当于没有找到。如果对红色框进行微调，让红框接近篮框。这个正是Bounding Box Regression 做的工作。

## 什么是边框回归，边框回归怎么做的

如何将红边框微调到蓝边框近似，偏移+缩放。边框可以用 中心点坐标，宽和高表示（x,y,w,h）。
<img src="{{ site.baseurl }}/img/2019-8-13-YOLOV3-STRUCTURE/yolo_v3_out.png" /> 
从上面的公式可以得到，根据模型输出值来移动中心坐标，和缩放宽和高。





