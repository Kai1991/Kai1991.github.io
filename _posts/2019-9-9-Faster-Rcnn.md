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
<img src="{{ site.baseurl }}/img/2019-9-9-Faster-Rcnn/fasterRCNN核心原理框图.jpg" /> 
这张图有两个线路：训练路线和推理路线。rpn（stage1） + detection target + fast rcnn (stage2) 为训练路线，rpn（stage1） + fast rcnn (stage2) 为推理路线。 下面详细介绍每个模块的功能。

## 训练数据-输入
训练数据的输入有：
- 图片，边框（真值） 
- 类别（所有类别中的一种）
- rpn_match(rpn推荐的anchor是否是前景)
- rpn_bboxes(前景anchor与真值的偏移量)

## 如何通过anchor得到真正的bboxes
 anchor 可以使用 中心点坐标，高和宽表示，如果调整anchor的到bboxes？显然是平移和缩放得到啦。对中心点进行平移，对高和宽进行缩放去逼近bboxes。应该平移多少，缩放多少这是模型学习的内容。


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
一开始学习时对anchor一直不是很理解，现在写一下我对anchor的理解：以特征图上的每个点为中心产生输入图像上的9个框。代码如下：
```
# 功能：构建 anchor
# featureMap_size=[8,8] 特征图大小
# ratios=[0.5, 1, 2]    宽高比
# scales=[4, 8, 16]     anchor的面积  
# rpn_stride=1          rpn的步长
# anchor_stride=1       anchor的步长             
def anchor_gen(featureMap_size, ratios, scales, rpn_stride, anchor_stride):
    #得到9个anchor
    ratios, scales = np.meshgrid(ratios, scales)
    ratios, scales = ratios.flatten(), scales.flatten()
    
    width = scales / np.sqrt(ratios)
    height = scales * np.sqrt(ratios)
    
    #得到特征图网格
    shift_x = np.arange(0, featureMap_size[0], anchor_stride) * rpn_stride
    shift_y = np.arange(0, featureMap_size[1], anchor_stride) * rpn_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    #获取每个特征图上点的9个anchor
    centerX, anchorX = np.meshgrid(shift_x, width)
    centerY, anchorY = np.meshgrid(shift_y, height)
    boxCenter = np.stack([centerY, centerX], axis=2).reshape(-1, 2)
    boxSize = np.stack([anchorX, anchorY], axis=2).reshape(-1, 2)
    
    #转换成 最小点 和最大点格式
    boxes = np.concatenate([boxCenter - 0.5 * boxSize, boxCenter + 0.5 * boxSize], axis=1)
    return boxes
```

### RPN-loss
rpn有两个损失函数：
- 分类损失。只使用iou大于0.7 和 iou小于 0.3的真值参与计算损失 使用交叉熵损失函数。
- 回归损失，使用的是smooth L1 loss(下雨阈值的使用二次函数，大于阈值的使用一次函数).
代码如下：
```
def rpn_class_loss(rpn_matchs,rpn_class_logits):
    # rpn_matchs 分类真值：这个anchor是否有 前景和背景以及中间项  起值对应 1，-1，0 ； shape (?,8*8*9,1)
    # rpn_logist rpn的预测值;  shape (?,8*8*9,2)

    rpn_matchs = tf.squeeze(rpn_matchs,axis=-1) # 压缩tensor翻遍去index
    indices = tf.where(tf.not_equal(rpn_matchs,0)) # 取出 1 和 -1 的框窜参与计算rpn分类
    anchor_class = K.cast(tf.equal(rpn_matchs,1),tf.int32) # 将非1的值转成0，前景为1 ，后景为0
    anchor_class = tf.gather_nd(anchor_class,indices) #target 

    rpn_class_logits = tf.gather_nd(rpn_class_logits,indices) #提取需要的预测值 # prediction

    loss = K.sparse_categorical_crossentropy(anchor_class,rpn_class_logits,from_logits=True)
    loss = K.switch(tf.size(loss) > 0,K.mean(loss),tf.constant(0.0)) #判断loss是否为零

    return loss

def batch_back(x,counts,num_rows):
    out_puts = []
    for i in range(num_rows):
        out_puts.append(x[i,:counts[i]])
    return tf.concat(out_puts,axis=0)
        

def rpn_bbox_loss(target_bbox,rpn_matchs,rpn_bbox):
    # target_bbox 目标框
    # rpn_matchs 真值是否有目标
    # rpn_bbox 预测框
    rpn_matchs = tf.squeeze(rpn_matchs,-1)
    indices = tf.where(K.equal(rpn_matchs,1))

    rpn_bbox = tf.gather_nd(rpn_bbox,indices)# 从预测框中提取对应位置的框

    batch_counts = K.sum(K.cast(K.equal(rpn_matchs,1),'int32'),axis=1) #统计每个图片中有几个bbox
    target_bbox = batch_back(target_bbox,batch_counts,20) #?
    diff  = K.abs(target_bbox - rpn_bbox)
    less_than_one = K.cast(K.less(diff,1),'float32')
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one)*(diff - 0.5) #?
    loss = K.switch(tf.size(loss) > 0,K.mean(loss),tf.constant(0.0))
    return loss

```

### DetectionTarget
此模块在训练时使用，为下层提供训练数据。筛选proposal提供的数据，筛选与真值iou高的数据推送数据
```
def detection_target_graph(proposals, gt_class_ids, gt_bboxes, config):
    #提取非0 部分：输入，ptoposal 为了固定长度使用 0进行padding
    proposals,_ = trim_zeros_graph(proposals,name='trim_proposals') 
    gt_bboxes,none_zeros = trim_zeros_graph(gt_bboxes,name='trim_bboxes')
    gt_class_ids = tf.boolean_mask(gt_class_ids,none_zeros)

    #计算每个proposal和每个gt_bboxes的iou 
    #加入有N个proposal 和 M个 gt_bboxes
    overlaps = overlaps_graph(proposals, gt_bboxes) #返回的shape：[N,M]
    max_iouArg = tf.reduce_max(overlaps,axis=1) # 沿着M压缩 取出N个最大值  用来判断哪个proposal 是前景
    max_iouGT = tf.argmax(overlaps,axis=0)# 沿着N压缩  计算出proposal 对应最适应的gt_bboxes

    positive_mask = max_iouArg > 0.5 #大于0.5的为前景
    positive_idxs = tf.where(positive_mask)[:,0] # 前景索引
    negative_idxs = tf.where(max_iouArg < 0.5)[:,0] # 背景索引


    num_positive = int(config.num_proposals_train *  config.num_proposals_ratio) #前景的数量
    positive_idxs = tf.random_shuffle(positive_idxs)[:num_positive]
    positive_idxs = tf.concat([positive_idxs, max_iouGT], axis=0)
    positive_idxs = tf.unique(positive_idxs)[0] # 前景索引
    
    num_positive = tf.shape(positive_idxs)[0] #前景的数量

    r = 1 / config.num_proposals_ratio
    num_negative = tf.cast(r * tf.cast(num_positive, tf.float32), tf.int32) - num_positive #背景的数量
    negative_idxs = tf.random_shuffle(negative_idxs)[:num_negative]#背景索引

    positive_rois = tf.gather(proposals,positive_idxs)
    negative_rois = tf.gather(proposals,negative_idxs)

    # 取出前景对应的gt_bbox
    positive_overlap = tf.gather(overlaps,positive_idxs)
    gt_assignment = tf.argmax(positive_overlap,axis=1)
    gt_bboxes = tf.gather(gt_bboxes,gt_assignment)
    gt_class_ids = tf.gather(gt_class_ids,gt_assignment)


    # 计算偏移量
    deltas = box_refinement_graph(positive_rois, gt_bboxes)
    deltas /= config.RPN_BBOX_STD_DEV # 算出来的太小，需要统一增大

    rois = tf.concat([positive_rois, negative_rois], axis=0)

    N = tf.shape(negative_rois)[0]
    P = config.num_proposals_train - tf.shape(rois)[0]
    
    rois = tf.pad(rois,[(0,P),(0,0)])
    gt_class_ids = tf.pad(gt_class_ids, [(0, N+P)])
    deltas = tf.pad(deltas,[(0,N+P),(0,0)])
    gt_bboxes = tf.pad(gt_bboxes,[(0,N+P),(0,0)])
    
    return rois, gt_class_ids, deltas, gt_bboxes
```

## Fast R-CNN

### ROIPooling

### fpn_classifiler

### fpn_loss

### DetectionLayer



