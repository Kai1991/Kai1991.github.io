---
layout:     post
title:      3D点云分割
subtitle:   点云初步认识
date:       2019-05-28
author:     Kai
header-img: img/home-bg-o.jpg
catalog: true
tags:
    - 点云
    - 分割
    - SqueezeSeg
---

## 前言
这节总结3D点云的分割方法-SqueezeSeg 

## 3D点云的缺点

* 无序性
    * 表现：点云的本质是一堆点，点的顺序不影响这些点在空间中对整体形状的表示。
    * 解决方法：使用对称函数（例如 maxpooling,sumpooling）提取特征，不会受到顺序的影响
* 旋转性
    * 表现：相同的点云在空间中经过一定的刚性变化（旋转或平移）坐标发生变化
    * 解决方法：STN（spacial transform network）

## 3D点云处理方式有很多，以下是不同的方式：
* pixel-based
    * 思路：把3D数据转化成2D数据（从不同角度对点云数据投影），然后在根据成熟的2D方法操作
* voxel-based
    * 思路：将点云划分成均匀的空间三维体素(体素网格提供了结构,体素数量像图片一样是不变的，也解决了顺序问题)
* tree-based
    * 使用tree来结构化点云 对稀疏点云进行高效地组织 再套用成熟的神经网络进行处理
* point-based
    * 直接对点云进行处理，使用对称函数解决点的无序性，使用空间变换解决旋转/平移性

## SqueezeSeg 模型总结
SqueezeSeg 模型的思路是把3D点云数据转化成2D信息，然后进行分割。

### 3D数据转2D
SqueezeSeg的做法是把点云数据投射到球面上，三地点云的点对应球面的点的关系，下图表示：
![关系](https://math.jianshu.com/math?formula=%5Ctheta%20%3D%20arcsin%5Cfrac%7Bz%7D%7B%5Csqrt%7Bx%5E2%2By%5E2%2Bz%5E2%7D%7D%2C%20%5Cwidetilde%7B%5Ctheta%7D%20%3D%20%5Clfloor%20%5Ctheta%20%2F%20%5CDelta%5Ctheta%20%5Crfloor%20%2C)
将等式（1）应用于云中的每个点，我们可以获得大小为H × W × C的3D张量。在本文中，我们考虑从具有64个垂直通道的Velodyne HDL-64E LiDAR收集的数据，因此H = 64。受KITTI数据集的数据注释的限制，我们只考虑90°的前视图区域并将其划分为512个网格所以W = 512。C是每个点的特征数。在我们的实验中，我们为每个点使用了5个特征：3个笛卡尔坐标（x，y，z），强度测量和范围。

### 网络结构
![模型结构](http://static.zybuluo.com/usiege/y1b2hheqi3kx18x3f5n4g92y/image_1cm2f0qar1p6qrj71m9n1h3f1vbj23.png)

代码：网络构造，重点阅读CRF模块

```shell
    #lidar_input点云输入数据:b*64*512*5
    conv1 = self._conv_layer(
        'conv1', self.lidar_input, filters=64, size=3, stride=2,
        padding='SAME', freeze=False, xavier=True) 
    conv1_skip = self._conv_layer(
        'conv1_skip', self.lidar_input, filters=64, size=1, stride=1,
        padding='SAME', freeze=False, xavier=True)
    pool1 = self._pooling_layer(
        'pool1', conv1, size=3, stride=2, padding='SAME')

    fire2 = self._fire_layer(
        'fire2', pool1, s1x1=16, e1x1=64, e3x3=64, freeze=False)
    fire3 = self._fire_layer(
        'fire3', fire2, s1x1=16, e1x1=64, e3x3=64, freeze=False)
    pool3 = self._pooling_layer(
        'pool3', fire3, size=3, stride=2, padding='SAME')

    fire4 = self._fire_layer(
        'fire4', pool3, s1x1=32, e1x1=128, e3x3=128, freeze=False)
    fire5 = self._fire_layer(
        'fire5', fire4, s1x1=32, e1x1=128, e3x3=128, freeze=False)
    pool5 = self._pooling_layer(
        'pool5', fire5, size=3, stride=2, padding='SAME')

    fire6 = self._fire_layer(
        'fire6', pool5, s1x1=48, e1x1=192, e3x3=192, freeze=False)
    fire7 = self._fire_layer(
        'fire7', fire6, s1x1=48, e1x1=192, e3x3=192, freeze=False)
    fire8 = self._fire_layer(
        'fire8', fire7, s1x1=64, e1x1=256, e3x3=256, freeze=False)
    fire9 = self._fire_layer(
        'fire9', fire8, s1x1=64, e1x1=256, e3x3=256, freeze=False)


    # Deconvolation
    fire10 = self._fire_deconv(
        'fire_deconv10', fire9, s1x1=64, e1x1=128, e3x3=128, factors=[1, 2],
        stddev=0.1)
    fire10_fuse = tf.add(fire10, fire5, name='fure10_fuse')

    fire11 = self._fire_deconv(
        'fire_deconv11', fire10_fuse, s1x1=32, e1x1=64, e3x3=64, factors=[1, 2],
        stddev=0.1)
    fire11_fuse = tf.add(fire11, fire3, name='fire11_fuse')

    fire12 = self._fire_deconv(
        'fire_deconv12', fire11_fuse, s1x1=16, e1x1=32, e3x3=32, factors=[1, 2],
        stddev=0.1)
    fire12_fuse = tf.add(fire12, conv1, name='fire12_fuse')

    fire13 = self._fire_deconv(
        'fire_deconv13', fire12_fuse, s1x1=16, e1x1=32, e3x3=32, factors=[1, 2],
        stddev=0.1)
    fire13_fuse = tf.add(fire13, conv1_skip, name='fire13_fuse')

    drop13 = tf.nn.dropout(fire13_fuse, self.keep_prob, name='drop13')

    conv14 = self._conv_layer(
        'conv14_prob', drop13, filters=mc.NUM_CLASS, size=3, stride=1,
        padding='SAME', relu=False, stddev=0.1) #shape:b*64*512*5


    bilateral_filter_weights = self._bilateral_filter_layer(
        'bilateral_filter', self.lidar_input[:, :, :, :3], # 初始化权重
        thetas=[mc.BILATERAL_THETA_A, mc.BILATERAL_THETA_R],
        sizes=[mc.LCN_HEIGHT, mc.LCN_WIDTH], stride=1)


    self.output_prob = self._recurrent_crf_layer(
        'recurrent_crf', conv14, bilateral_filter_weights, 
        sizes=[mc.LCN_HEIGHT, mc.LCN_WIDTH], num_iterations=mc.RCRF_ITER,
        padding='SAME') 

```

### CRF
通俗理解CRF，前面模型做分割对于细节不太友好，需要用工具进一步优化细节，而这个工具就是CRF。CRF的原理是这样的，像素的点属于哪个类别，可以根据上下文（像素周围的像素点，再牛叉点其他所有的点）来进一步推理判断是不是属于这一类。

![CRF部分结构图](http://static.zybuluo.com/usiege/ao2ra6h2zt1w9ds95orjynxa/image_1cm4t2nj82ff1oee18tc1o3k17ac16.png)

```shell
  def _recurrent_crf_layer(
      self, layer_name, inputs, bilateral_filters, sizes=[3, 5],
      num_iterations=1, padding='SAME'):

    mc = self.mc

    with tf.variable_scope(layer_name) as scope:
      # initialize compatibilty matrices
      compat_kernel_init = tf.constant(
          np.reshape(
              np.ones((mc.NUM_CLASS, mc.NUM_CLASS)) - np.identity(mc.NUM_CLASS),
              [1, 1, mc.NUM_CLASS, mc.NUM_CLASS]
          ),
          dtype=tf.float32
      )
      #构造变量权重
      bi_compat_kernel = _variable_on_device(
          name='bilateral_compatibility_matrix',
          shape=[1, 1, mc.NUM_CLASS, mc.NUM_CLASS],
          initializer=compat_kernel_init*mc.BI_FILTER_COEF,
          trainable=True
      ) 
      angular_compat_kernel = _variable_on_device(
          name='angular_compatibility_matrix',
          shape=[1, 1, mc.NUM_CLASS, mc.NUM_CLASS],
          initializer=compat_kernel_init*mc.ANG_FILTER_COEF,
          trainable=True
      ) 
      #构造常量 用于消息传递
      condensing_kernel = tf.constant(
          util.condensing_matrix(sizes[0], sizes[1], mc.NUM_CLASS),
          dtype=tf.float32,
          name='condensing_kernel'
      ) #shape:3*5*4*(3*5*4 - 1) 
      #固定的录波器
      angular_filters = tf.constant(
          util.angular_filter_kernel(
              sizes[0], sizes[1], mc.NUM_CLASS, mc.ANG_THETA_A**2),
          dtype=tf.float32,
          name='angular_kernel'
      )#shape:3*5*4*4 
      bi_angular_filters = tf.constant(
          util.angular_filter_kernel(
              sizes[0], sizes[1], mc.NUM_CLASS, mc.BILATERAL_THETA_A**2),
          dtype=tf.float32,
          name='bi_angular_kernel'
      )#shape:3*5*4*4 
      #循环处理
      for it in range(num_iterations):
        unary = tf.nn.softmax(
            inputs, dim=-1, name='unary_term_at_iter_{}'.format(it)) # 一元势函数
        #消息传递
        ang_output, bi_output = self._locally_connected_layer(
            'message_passing_iter_{}'.format(it), unary,
            bilateral_filters, angular_filters, bi_angular_filters,
            condensing_kernel, sizes=sizes,
            padding=padding
        )

        # 转换
        ang_output = tf.nn.conv2d(
            ang_output, angular_compat_kernel, strides=[1, 1, 1, 1],
            padding='SAME', name='angular_compatibility_transformation')

        self._activation_summary(
            ang_output, 'ang_transfer_iter_{}'.format(it))

        bi_output = tf.nn.conv2d(
            bi_output, bi_compat_kernel, strides=[1, 1, 1, 1], padding='SAME',
            name='bilateral_compatibility_transformation')

        pairwise = tf.add(ang_output, bi_output,
                          name='pairwise_term_at_iter_{}'.format(it))#二元势函数

        outputs = tf.add(unary, pairwise,
                         name='energy_at_iter_{}'.format(it))

        inputs = outputs

    return outputs

```
注：部分不重要代码有删除

## 引用
* [SqueezeSeg: Convolutional Neural Nets with Recurrent CRF for Real-Time Road-Object Segmentation f...](https://www.jianshu.com/p/ade8108fe370)
* [Conditional Random Fields as Recurrent Neural Networks](https://blog.csdn.net/amds123/article/details/69568590)








