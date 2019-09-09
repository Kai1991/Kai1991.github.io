---
layout:     post
title:      faster_rcnn总结
subtitle:   Faster_rcnn
date:       2019-09-09
author:     Kai
header-img: img/home-bg-o.jpg
catalog: true
tags:
    -  Faster_rcnn 
    -  检测
---

## 前言

Faster R-cnn代码实现，原理记录。分模块介绍Faster R-cnn，它有如下模块：
- 特征提取模块, 使用Resnet50来提取
- rpn模块.     rpn,Region Proposal,rpn中的两个loss（分类loss，回归loss）,detection target(不会学习，这是为后面处理数据)
- fpn模块.     Roi pooling,fpn_classifiler 分类回归,fpn中的两个loss

现在首先大体介绍一下Faster R-cnn 。来张整体结构图，对她有个大体了解：
<img src="{{ site.baseurl }}img/2019-9-9-Faster-Rcnn/fasterRCNN 核心原理框图.jpg" /> 
这张图有两个线路：训练路线和推理路线。rpn（stage1） + detection target + fast rcnn (stage2) 为训练路线，rpn（stage1） + fast rcnn (stage2) 为推理路线。 下面详细介绍每个模块的功能。

## Feature Extractor
特征提取模块使用的是 Resnet50 来提取特征，感觉没啥好描述的。简单描述一下：下采样的步长（TODO）得到的特征图是（TODO）。

## RPN
经过一层卷积，然后获取 分类（前景，背景）和回归。代码如下：

```
def rpn_net(inputs,k=9):
    #构建rpn网络

    # inputs : 特证图 shape(batch_size,8,8,1024)
    # k : 特证图上anchor的个数

    # 返回值： 
    #   rpn分类
    #   rpn分类概率
    #   rpn回归
    shared_map = KL.Conv2D(256,(3,3),padding='same')(inputs) #shape(batch_size,8,8,256)
    shared_map = KL.Activation("linear")(shared_map)

    rpn_class = KL.Conv2D(2*k,(1,1))(shared_map) #shape(batch_size,8,8,2*k)
    rpn_class = KL.Lambda(lambda x : tf.reshape(x,(tf.shape(x)[0],-1,2)))(rpn_class)
    rpn_class = KL.Activation('linear')(rpn_class) #shape(batch_size,8*8*k,2)
    
    rpn_prob = KL.Activation('softmax')(rpn_class)

    rpn_bbox = KL.Conv2D(4*k,(1,1))(shared_map) #shape(batch_size,8,8,4*k)
    rpn_bbox = KL.Activation('linear')(rpn_bbox)
    rpn_bbox = KL.Lambda(lambda x : tf.reshape(x,(tf.shape(x)[0],-1,4)))(rpn_bbox) #shape(batch_size,8*8*k,4) 8*8*9 = 576

    return rpn_class,rpn_prob,rpn_bbox
```
### Proposal
根据RPN中得到的分类和回归 和anchor 来筛选哪个anchor合适。至于anchor的生成下个章节单独说明。首先提取topN(N=100,可以设定)的数据用来proposal。然后根据rpn的到的偏移量来调节anchor(移动，缩放),然后根据对边框进行 归一化 和最大值抑制操作。代码如下：
```
class proposal(KE.Layer):
    def __init__(self,proposal_count,nms_thresh,anchors,batch_size,config,**kwargs):
        super(proposal,self).__init__(**kwargs)
        self.proposal_count = proposal_count
        self.anchors = anchors
        self.nms_thresh = nms_thresh
        self.batch_size = batch_size
        self.config = config
    def call(self,inputs):
        probs = inputs[0][:,:,1] #shape(batch_size,576,1)
        deltas = inputs[1] #shape(batch_size,576,4)   
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV,(1,1,4)) #denormalization
        prenms = min(100,self.anchors.shape[0]) #最多取出100个
        print("prenms:")
        print(prenms)
        idxs = tf.nn.top_k(probs,prenms).indices # 钱top

        #提取相关数据
        probs = batch_slice([probs,idxs],lambda x,y :tf.gather(x,y),self.batch_size)
        deltas = batch_slice([deltas,idxs],lambda x,y :tf.gather(x,y),self.batch_size)
        anchors = batch_slice([idxs],lambda x : tf.gather(self.anchors,x),self.batch_size) #批次内对应的每组anchor

        refined_boxes = batch_slice([anchors,deltas],lambda x,y:anchor_refinement(x,y),self.batch_size) #调整anchor

        #防止 proposal 的框超出图片区域，剪切一下
        H,W = self.config.image_size[:2]
        windows = np.array([0,0,H,W]).astype(np.float32)
        cliped_boxes = batch_slice([refined_boxes], lambda x: boxes_clip(x,windows),self.batch_size)

        # 对proposal进行归一化  使用的是图片大小进行归一化的
        normalized_boxes = cliped_boxes / tf.constant([H,W,H,W],dtype=tf.float32)  #这里不一样

        def nms(normalized_boxes, scores):
            idxs_ = tf.image.non_max_suppression(normalized_boxes,scores,self.proposal_count,self.nms_thresh)
            box = tf.gather(normalized_boxes,idxs_)
            pad_num = tf.maximum(self.proposal_count - tf.shape(box)[0],0)
            box = tf.pad(box,[(0,pad_num),(0,0)])# 填充0
            return box
        # 对proposal进行nms 最大值抑制
        proposal_ = batch_slice([normalized_boxes,probs],lambda x,y : nms(x,y),self.batch_size)
        return proposal_ 
  
        
    def compute_output_shape(self,input_shape):
        return (None,self.proposal_count,4)
```

### Anchor
一开始学习时对anchor一直不是很理解，现在写一下我对anchor的理解。

### RPN-loss

### DetectionTarget

## Fast R-CNN

### ROIPooling

### fpn_classifiler

### fpn_loss

### DetectionLayer



