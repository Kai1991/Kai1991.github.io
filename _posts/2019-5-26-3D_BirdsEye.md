---
layout:     post
title:      3D点云转化鸟瞰图
subtitle:   点云初步认识
date:       2019-05-26
author:     Kai
header-img: img/home-bg-o.jpg
catalog: true
tags:
    - 点云
---

## 前言

上一篇论文初步了解了点云，这一篇动手实现3D点云转化成鸟瞰图，感受感受一下。


## 代码

```shell
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 转换成 像素值
def scale_2_255(a,min,max,dtype=np.uint8):

    return ((a - min) / float(max - min)  * 255 ).astype(dtype)

def point_clound_2_birdseye(points,res=0.1,
                            side_range = (-10.,10),
                            fwd_range = (-10.,10),
                            height_range = (-2.,0.5)):
    # 分别获取3个维度的数据
    x_points = points[:,0]
    y_points = points[:,1]
    z_points = points[:,2]

    #获取目标范围的mask
    f_filt = np.logical_and((x_points > fwd_range[0]),(x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > -side_range[1]),(y_points < -side_range[0]))
    filt = np.logical_and(f_filt,s_filt)
    indices = np.argwhere(filt).flatten()

    #去除目标点
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    # 转化成图像像素点 由于坐标不一样
    x_img = (-y_points / res).astype(np.int32)
    y_img = (-x_points / res).astype(np.int32)

    # 调整坐标原点
    x_img -= int(np.floor(side_range[0]) / res)
    y_img += int(np.floor(fwd_range[1]) / res)
    print(x_img.min(), x_img.max(), y_img.min(), x_img.max())


    pixel_values = np.clip(a=z_points,
                            a_min=height_range[0],
                            a_max=height_range[1])
    
    pixel_values = scale_2_255(pixel_values,min=height_range[0],
                                max=height_range[1])

    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max,x_max],dtype=np.uint8)

    #填充像素值
    im[y_img,x_img] = pixel_values

    return im

def main():
    #读取点云数
    pointcloud = np.fromfile(str("./data/0000000001.bin"), dtype=np.float32, count=-1).reshape([-1, 4])
    #将点云数据投射鸟瞰图
    img = point_clound_2_birdseye(pointcloud)

    #图片展示
    im2 = Image.fromarray(img)
    im2.show()
    #plt.imshow(img, cmap="spectral", vmin=0, vmax=255)
    #plt.show()


if __name__ == "__main__":
    main()

```
* 结果
![结果](http://ronny.rest/media/tutorials/lidar/point_cloud_birdseye/birdseye_pil.jpg)

[git地址](https://github.com/Kai1991/bird_eye_pointclound.git)


## 引用
[Processing Point Clouds](http://ronny.rest/tutorials/module/pointclouds_01)








