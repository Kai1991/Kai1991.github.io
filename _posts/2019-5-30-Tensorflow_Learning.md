---
layout:     post
title:      Tensorflow学习笔记
subtitle:   学习记录
date:       2019-05-30
author:     Kai
header-img: img/home-bg-o.jpg
catalog: true
tags:
    - Tensorflow
---

## 前言
Tensorflow学习使用记录

## Tensorflow 操作算子

### tf.squeeze 压缩
压缩tensor,把shape中为1的干掉，可以指定具体哪个秩。

```shell
import tensorflow as tf

random_int_var_one_ex = tf.get_variable("random_int_var_one_ex", [1, 2, 3], dtype=tf.int32,
  initializer=tf.zeros_initializer)
with tf.Session() as sess:
    #初始化变量
    sess.run(tf.global_variables_initializer())
    #执行获取变量算子
    print(sess.run(random_int_var_one_ex))
    #打印没有变化前的形状
    print(sess.run(tf.shape(random_int_var_one_ex)))
    #变化
    squeezed_tensor_ex = tf.squeeze(random_int_var_one_ex)
    sess.run(squeezed_tensor_ex)
    #打印变化后的形状
    print(sess.run(tf.shape(squeezed_tensor_ex)))
    print(sess.run(squeezed_tensor_ex))
```
结果：
```shell
[[[0 0 0]
  [0 0 0]]]

[1 2 3]

[2 3]

[[0 0 0]
 [0 0 0]]
```

### tf.scatter_nd

把目标tensor切片，插入到另一个指定形状的tensor（值全为0）中.
```shell
import tensorflow as tf

indices = tf.constant([[0],[2]])
updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                        [7, 7, 7, 7], [8, 8, 8, 8]],
                       [[5, 5, 5, 5], [6, 6, 6, 6],
                        [7, 7, 7, 7], [8, 8, 8, 8]]]) # shape:2*4*4

shape = tf.constant([4,4,4])
scatter = tf.scatter_nd(indices, updates, shape)
with tf.Session() as sess:
    print(sess.run(scatter))

```
过程
<img src="{{ site.baseurl }}/img/2019-5-30-TensorFlow-Learning/tf.scatter_nd.png" />
结果：
```shell
[[[5 5 5 5]
  [6 6 6 6]
  [7 7 7 7]
  [8 8 8 8]]

 [[0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]]

 [[5 5 5 5]
  [6 6 6 6]
  [7 7 7 7]
  [8 8 8 8]]

 [[0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]]]
```
注：引用 [TensorFlow学习（三）：tf.scatter_nd函数](https://blog.csdn.net/zlrai5895/article/details/80551056)

### tf.tile()
对 tensor(张量) 按维度进行复制扩充
```shell
import tensorflow as tf

tmp = tf.constant([[1,2,3]]) # shape:1*3
tile_op = tf.tile(tmp,[2,3]) # 处理后的shape:2*9
with tf.Session() as sess:
    print('处理前：')
    print(sess.run(tmp))
    print('处理后：')
    print(sess.run(tile_op))
```
过程：
<img src="{{ site.baseurl }}/img/2019-5-30-TensorFlow-Learning/tf.tile.png" />
结果：
```shell
处理前：
[[1 2 3]]
处理后：
[[1 2 3 1 2 3 1 2 3]
 [1 2 3 1 2 3 1 2 3]]
```
注：引用 [直观的理解tensorflow中的tf.tile()函数](https://blog.csdn.net/tsyccnh/article/details/82459859),博客中表述有点问题，只使用了他的图。




