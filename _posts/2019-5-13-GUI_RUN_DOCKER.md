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

学习3D点云想要用到ros的rviz。一开始要用Mac装ROS，没有成功（非常蛋痛，放弃了），选择使用docker。官方支持非常好，官方的docker没有安装rviz。需要自己安装

## 安装 socat

```shell
brew install socat
socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\" 
```

## 安装 xquartz

```shell
brew install xquartz
```

#### 运行 xquartz

```shell
open -a Xquartz
```

#### 配置 xquartz

在 xquartz 个性化设置的安全 勾选 ‘允许网络端链接’

## 运行容器 

```shell
docker run -e DISPLAY=192.168.64.112:0 -it ros_rviz:latest
```
备注： 192.168.64.112 这里的IP改成自己宿主机的ip。ros_rviz:latest ：快照名：版本号

## 在容器内启动 rviz

```shell
rosrun rviz rviz
```


## 引用
[讲的很好，如果打不开的话，请科学上网，^……^](https://cntnr.io/running-guis-with-docker-on-mac-os-x-a14df6a76efc)




