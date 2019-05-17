---
layout:     post
title:      在Docker上运行GUI程序
subtitle:   运行rviz为例-Mac
date:       2019-05-13
author:     Kai
header-img: img/home-bg-o.jpg
catalog: true
tags:
    - docker
    - rviz
---

## 前言

学习3D点云想要用到ros的rviz。一开始要用Mac装ROS，没有成功（非常蛋痛，放弃了），选择使用docker。官方支持非常好，官方的docker没有安装rviz。需要自己安装 .ros官方的docker安装完后，依然有问题，并且不好解决。下面介绍一种非常简洁，傻瓜式安装（全网最简洁方式）。

## 拉取别人做好的镜像

```shell
docker pull ct2034/vnc-ros-kinetic-full
```

## 运行 容器

```shell
docker run -it --rm -p 6080:80 ct2034/vnc-ros-kinetic-full
```

## 浏览器 输入地址

```shell
http://127.0.0.1:6080/
```


## 在容器内启动主节点 

```shell
roscore
```
## 在容器内启动 rviz

```shell
rosrun rviz rviz
``` 

## 结果
看到这个结果想哭，折腾了好长时间才找到这么方便的方式。。。。。
![](https://raw.github.com/ct2034/docker-ubuntu-vnc-desktop/master/screenshots/ros-kinetic.png)


## 引用
[docker 镜像作者，里面也有说明](https://hub.docker.com/r/ct2034/vnc-ros-kinetic-full/)




