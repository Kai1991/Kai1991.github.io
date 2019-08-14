---
layout:     post
title:      YOLO_V3-结构篇
subtitle:   学习记录
date:       2019-08-13
author:     Kai
header-img: img/home-bg-o.jpg
catalog: true
tags:
    - YOLO_V3
---

## 前言
Keras-TOLO3结构学习总结，分别从：输入，结构，输出 三方面说明YOLO_V3。看的是qwe大神的代码[YOLO_V3](https://github.com/qqwweee/keras-yolo3)

## 模型输入
- 首先，训练自己的数据，图片大小不需要处理。因为代码里已经处理压缩了
```shell
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.
```
- 其次，输入处理主要做的工作是：
    - 增强数据
    - 选择真值（边框）对应适合的anchor.计算真值和每个anchor的IOU,选择合适的anchor。注意：训练时的真值是处理过的，中心点，高和宽  都是 ➗ 输入大小（不是图片）✖️ anchor网格大小
```shell
    def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)  真正的边框
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32   输入模型的形状
    anchors: array, shape=(N, 2), wh   
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting   输出层数
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # anchor 分类

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2 # 中心点坐标
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]        # 边框宽和高
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1] # 对中心点使用输入大小进行归一化
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1] # 对边框宽和高使用输入大小进行归一化

    m = true_boxes.shape[0] #边框的最大数量 
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)] # 三层输出大小，分别是:输入的大小//32,输入的大小//16,输入的大小//8
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0) # shape (9,2) -> shape (1,9,2)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0 #有边框的mask

    for b in range(m): # 遍历每个边框
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue # 剔除没有边框的情况
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2) # ？有疑问
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box 找到对于真值来说最适合的anchor
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):#输出层数
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true
```

## 结构
<img src="{{ site.baseurl }}/img/2019-8-13-YOLOV3-STRUCTURE/yolo_v3.png" /> 
注： 图片引用这篇[博客](https://blog.csdn.net/leviopku/article/details/82660381)

主要组件：
- DBL : 主要是 卷积+BN+Leaky relu 的组合。 
- resn : n代表数字，有res1，res2, … ,res8等等，表示这个res_block里含有多少个res_unit。yolo_v3开始借鉴了ResNet的残差结构，使用这种结构可以让网络结构更深
- concat :  张量拼接。将darknet中间层和后面的某一层的上采样进行拼接。拼接的操作和残差层add的操作是不一样的，拼接会扩充张量的维度，而add只是直接相加不会导致张量维度的改变。 这样做的目的是：可以感受大小不同的感受野。

## 输出
<img src="{{ site.baseurl }}/img/2019-8-13-YOLOV3-STRUCTURE/yolo_v3_out.png" /> 
注 
- (t_x, t_y , t_w , t_h , t_o) 是模型的输出
- 然后通过公式1计算出绝对的(x, y, w, h, c)

## 损失函数

在目标检测任务里，有几个关键信息是需要确定的: 中心坐标（x,y），边框宽高（w,h）,置信度，分类。根据关键信息的特点可以分为上述四类，损失函数应该由各自特点确定。最后加到一起就可以组成最终的loss_function了，也就是一个loss_function搞定端到端的训练。

```shell
xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                       from_logits=True)
wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                          (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                    from_logits=True) * ignore_mask
class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

xy_loss = K.sum(xy_loss) / mf
wh_loss = K.sum(wh_loss) / mf
confidence_loss = K.sum(confidence_loss) / mf
class_loss = K.sum(class_loss) / mf
loss += xy_loss + wh_loss + confidence_loss + class_loss
```
