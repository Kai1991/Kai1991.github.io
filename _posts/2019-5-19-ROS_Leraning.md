---
layout:     post
title:      ROS基础学习
subtitle:   ROS基础
date:       2019-05-19
author:     Kai
header-img: img/home-bg-o.jpg
catalog: true
tags:
    - ROS
---

## 前言

ROS的一些基础知识

## ROS的组成部分

## ROS 命令

* roscore: 启动主节点
* rosrun 报名 节点名： 启动节点
* rospack list:查看包
* rosnode list:查看节点
* rostopic list：查看主题

## catkin 使用

* catkin_create_pkg : 创建应用包   . 例子： catkin_create_pkg test rosapp （ 创建test包 同时引用rosapp包）

* catkin_make : 编译包 注：要在工作空间目录运行 ，同时编译前需要执行：source ./devel/setup.bash




