---
layout:     post
title:      Kalman Filter学习笔记
subtitle:   学习记录
date:       2019-06-06
author:     Kai
header-img: img/home-bg-o.jpg
catalog: true
tags:
    - Kalman Filter
---

## 前言
Kalman Filter 使用场景非常多，比如 追踪，定位，数据融合。虽然叫卡尔曼录波器，但是他不是滤波器。它是数据融合的工具。
## 栗子

一个小车在行驶，此时是t时刻我们想知道这个车子的状态（车子的位置，车子的速度，，，），为了得到这些信息有两个手段，手段一：我们知道上一时刻的状态我们可以预测出来这一时刻的状态（有误差），手段二：我们可以使用设备去观测状态（也有误差）。如何使用这两部分信息。如何进行融合。此时卡尔曼滤波排上用场了。

## 抽象

- 小车t时刻的状态：所在位置pt,速度vt
- 小车在t时刻的控制变量：司机可以踩油门加速，加速度是ut

运动状态方程可以表示成如下：

- t时刻的位子可以从上一时刻预测得到： 
<img src="{{ site.baseurl }}/img/2019-6-6-KalmanFilter/fun1.png" />
- 使用矩阵表示上述公式
<img src="{{ site.baseurl }}/img/2019-6-6-KalmanFilter/fun2.png" />
- 进一步抽象
<img src="{{ site.baseurl }}/img/2019-6-6-KalmanFilter/fun3.png" />
<img src="{{ site.baseurl }}/img/2019-6-6-KalmanFilter/fun4.png" />

## 运动方程
<img src="{{ site.baseurl }}/img/2019-6-6-KalmanFilter/fun6.png" />
注：A是状态转换矩阵，B是控制矩阵

## 观测方程
真是情况我们是无法得到的（测试仪器也是有误差的），假设路边设置一个观测器，观测到的数据是 Z，则真是
<img src="{{ site.baseurl }}/img/2019-6-6-KalmanFilter/fun7.png" />
注：H是观测转换矩阵 V是观测的误差，符合高斯分布

## Kalman Filter 的思想
根据预测值和观察值的误差来确定哪部分的数据置信度高，根据这个置信度来融合数据，来预估状态。

## 完整公式
<img src="{{ site.baseurl }}/img/2019-6-6-KalmanFilter/fun8.png" />

## 需要用到的知识

- 协方差：两个变量的总体的误差。
- 协方差公式：<img src="{{ site.baseurl }}/img/2019-6-6-KalmanFilter/fun5.png" /> 可以从公式可得，协方差是两个变量总体误差的期望。
- 协方差矩阵：协方差矩阵的每个元素是各个向量元素之间的协方差




