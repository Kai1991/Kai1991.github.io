---
layout:     post
title:      PointNet
subtitle:   解读PointNet代码&原理
date:       2019-04-29
author:     Kai
header-img: img/home-bg-o.jpg
catalog: true
tags:
    - 自动驾驶
    - 感知
    - PointNet系列
    - 3D点云
    - 3D点云分类
    - 3D点云分割
---

## 前言

想学习一下3D点云的分类，分割和检测的技术.PointNet代码简洁，官方对于代码复现做的也很好，非常适合拿来学习。此博客记录自己对PointNet的理解。

## 使用PointNet实现分类

### 使用到的分类训练，验证数据

分类数据 [] (https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)

### 部分代码修改

我习惯把训练验证数据和代码分开，建议在代码文件同一级别建 data文件夹存放数据，所以训练数据的地址进行修改

```
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, '../data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, '../data/modelnet40_ply_hdf5_2048/test_files.txt'))
```

数据也需要修改，修改如下
```
../data/modelnet40_ply_hdf5_2048/ply_data_train0.h5
../data/modelnet40_ply_hdf5_2048/ply_data_train1.h5
../data/modelnet40_ply_hdf5_2048/ply_data_train2.h5
../data/modelnet40_ply_hdf5_2048/ply_data_train3.h5
../data/modelnet40_ply_hdf5_2048/ply_data_train4.h5
```

然后执行

```shell
nohup python train.py & 
```

训练结果
![](https://github.com/Kai1991/Kai1991.github.io/blob/master/img/2019-4-28-PointNet/class_result.jpg)

## PonitNet分类代码走读

这里有三个方法需要认真读一下，分别是：get_model, input_transform_net, feature_transform_net

### PointNet模型结构

![](https://github.com/Kai1991/Kai1991.github.io/blob/master/img/2019-4-28-PointNet/pointNet_structure.jpg)

### get_model 方法解读，看注解

```
def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    with tf.variable_scope('transform_net1') as sc:
        # 这里获取 3*3的转换矩阵
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3) # transform shape : B*3*3
    # 通过转换矩阵调整视角，对其坐标
    point_cloud_transformed = tf.matmul(point_cloud, transform) # shape : B*N*3 
    input_image = tf.expand_dims(point_cloud_transformed, -1) # 扩展一个维度 shape : B*N*3*1

    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay) # shape :  B*N*1*64
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)# shape :  B*N*1*64

    with tf.variable_scope('transform_net2') as sc:
        #获取 特征的转换矩阵 
        transform = feature_transform_net(net, is_training, bn_decay, K=64) # shape: B*64*64
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform) # shape :  B*N*64
    net_transformed = tf.expand_dims(net_transformed, [2]) # shape :  B*N*1*64

    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)# shape :  B*N*1*64
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)# shape :  B*N*1*128
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)# shape :  B*N*1*1024

    # Symmetric function: max pooling 这里很重要，这里就是 对称函数
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool') # shape :  B*1*1*1024

    net = tf.reshape(net, [batch_size, -1]) # shape: B * 1024
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay) # shape: B * 512
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay) # shape: B * 256
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3') # shape: B * 40

    return net, end_points
```

### input_transform_net 方法

```
def input_transform_net(point_cloud, is_training, bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    input_image = tf.expand_dims(point_cloud, -1)  #-1表示最后一维  shape B * N * 3 * 1
    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay) # shape : B * N * 1 * 64
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay) # shape : B * N * 1 * 128
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay) # shape : B * N * 1 * 1024
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool') # shape : B * 1 * 1 * 1024

    net = tf.reshape(net, [batch_size, -1])  # shape : B * 1024
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay) # shape : B * 512
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay) # shape : B * 256

    with tf.variable_scope('transform_XYZ') as sc:
        assert(K==3)
        weights = tf.get_variable('weights', [256, 3*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [3*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
        transform = tf.matmul(net, weights) # shape : B * 3k
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, 3, K]) # shape : B * 3 * k
    return transform
```

> 对这个转换网络的生成还是不太懂，后面明白了再记录，[这里讲Spatial Transformer Networks](https://blog.csdn.net/qq_39422642/article/details/78870629)讲的非常好，通俗易懂。


### feature_transform_net 方法

```
def feature_transform_net(inputs, is_training, bn_decay=None, K=64):
    """ Feature Transform Net, input is BxNx1xK
        Return:
            Transformation matrix of size KxK """
    batch_size = inputs.get_shape()[0].value
    num_point = inputs.get_shape()[1].value

    net = tf_util.conv2d(inputs, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_feat') as sc:
        weights = tf.get_variable('weights', [256, K*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [K*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform
```
