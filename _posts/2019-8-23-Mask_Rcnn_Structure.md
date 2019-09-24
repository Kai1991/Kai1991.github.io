---
layout:     post
title:      Mask_rcnn结构源码总结之anchor的生成
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

## 前言
这篇只写mask-rcnn锚定框的生成

## anchor的生成

由于Mask-rcnn的backbone使用的是Resnet50(101) + FPN ,可想而知每层特征图对应的anchor的感受野不同，低层的特征anchor的感受野应该比较小，每层有三个anchor，感受野一样，只是宽高比不同。

Mask-rcnn与YOLO_v3的anchor很像，不过yolo_v3的通过Kmeans聚类得到的每层有三个，低层的anchor的感受野要小。

mask-rcnn的anchor是一特征图的像素点为中心.下面看一下anchor的生成代码：

anchor生成的入口函数是 get_anchors(self, image_shape)
```
def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape) #计算每层特征图的大小 如果image_shape=[256,256] [(64,64),(32,32),(16,16),(4,4),(2,2)]
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES, #(32, 64, 128, 256, 512) 每层anchor的感受野
                self.config.RPN_ANCHOR_RATIOS, # [0.5, 1, 2] 宽高比
                backbone_shapes, # [256,256] [(64,64),(32,32),(16,16),(4,4),(2,2)] 每层特征图的大小
                self.config.BACKBONE_STRIDES, # [4, 8, 16, 32, 64] 特征图的步长
                self.config.RPN_ANCHOR_STRIDE) # 1 anchor的步长
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]
```
主要的anchor的生成函数在这里utils.generate_pyramid_anchors,此函数包装了anchor的生成

```
def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride)) # 可以从这里看出每层使用一个感受野，都使用相同的宽高比
    return np.concatenate(anchors, axis=0)
```
函数调了这个函数generate_anchors
```
def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    #根据面积和宽高比 构建anchor
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios)) # 构建网格，
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios) 
    # 根据 特征图构建 特征图网格
    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # 拼接特征图和anchor,以特征图的像素为中心，生成3个anchor
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # 装换格式成（y_min,x_min,y_max,x_max）
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes
```

anchor的生成结果:
<img src="{{ site.baseurl }}/img/2019-8-23-Mask_Rcnn_Structure/mask-rcnn-anchor.jpg" /> 



