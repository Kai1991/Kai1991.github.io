---
layout:     post
title:      Mask_rcnn结构源码总结之ROiAlign
subtitle:   Mask_rcnn
date:       2019-08-23
author:     Kai
header-img: img/home-bg-o.jpg
catalog: true
tags:
    -  Mask_rcnn 
    -  分割
    -  检测
---

<img src="{{ site.baseurl }}/img/2019-9-24-Mask_Rcnn_ROIAlign/mask-rcnn.jpg" /> 

## 前言
这篇只写ROiAlign。
ROiAlign是Mask-Rcnn相比Faster-rcnn的一个创新点，为什么提出ROiAlign，因为RoiPooling不好啊！接下来说一下RoiPooling的缺点。

## RoiPooling的缺点
ROI Pooling操作中两次量化造成的区域不匹配(mis-alignment)的问题。
<img src="{{ site.baseurl }}/img/2019-9-24-Mask_Rcnn_ROIAlign/roi_pooling_error.jpg" /> 

通过上图可以看出，假如图片的大小是800*800，bbox是665*665
- 经过卷积下采样边框成为 665/32 = 20.78  取整 20
- 取出固定大小7*7  20/7= 2.86 取整 2

## ROIAlign操作过程

一图胜千言
<img src="{{ site.baseurl }}/img/2019-9-24-Mask_Rcnn_ROIAlign/RoiAlign.jpg" /> 

通过上图可以看出，假如图片的大小是800*800，bbox是665*665
- 经过卷积下采样边框成为 665/32 = 20.78  不取整
- 取出固定大小7*7  20.78/7= 2.97 不取整
- 看下图，为了把bbox的特征图映射成7*7，需要把2.97*2.97的框均分成4份，每一份通过[双线性插值](https://blog.csdn.net/qq_37577735/article/details/80041586)得到一个值，一共得到4个值，然后对着四个值进行maxpooling得到一个值.

<img src="{{ site.baseurl }}/img/2019-9-24-Mask_Rcnn_ROIAlign/双线性插值.jpg" /> 


## 代码
由于 
backbone使用的是resnet + fpn 所以还有一个问题需要处理：proposal对应那一层特征图。
mask-rcnn通过proposal 的面积大小来映射这个proposal属于那一层的,公式如下：
<img src="{{ site.baseurl }}/img/2019-9-24-Mask_Rcnn_ROIAlign/roi_level.jpg" /> 
 k:是计算出来roi对应的层数，k0:是4，w,h=224对应的层数

roi_level 存储的roi属于哪一层
```

class PyramidROIAlign(KE.Layer):

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        #提案
        boxes = inputs[0]
        #图片信息
        image_meta = inputs[1]
        # 5层特征图 P2-P6 
        feature_maps = inputs[2:]

        #计算提案的大小
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1

        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]

        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area))) #计算公式
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)


            box_indices = tf.cast(ix[:, 0], tf.int32)

            box_to_level.append(ix)


            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))


        pooled = tf.concat(pooled, axis=0)


        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )
```
