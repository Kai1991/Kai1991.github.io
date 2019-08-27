---
layout:     post
title:      YOLO_V3-训练篇
subtitle:   YOLO_V3训练
date:       2019-08-15
author:     Kai
header-img: img/home-bg-o.jpg
catalog: true
tags:
    - YOLO_V3 training 
---

## 前言
使用的天池的比赛数据集[2019广东工业智造创新大赛-赛场一：布匹疵点智能识别](https://tianchi.aliyun.com/competition/entrance/231748/introduction) 练手YOLO_v3

## 数据输入
训练数据格式
```shell
数据格式： image_file_path box1 box2 ... boxN
box的格式：x_min,y_min,x_max,y_max,class_id
```

根据数据进行通过k-means获取anchors

## 模型输出

- box的输出格式是：（top, left, bottom, right ） 或者这种格式  (min_y,min_x,max_y,max_x)  没有注意这个，被害😢了。







