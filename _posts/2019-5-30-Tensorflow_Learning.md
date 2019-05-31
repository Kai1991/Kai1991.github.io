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

* tf.squeeze 压缩
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

* tf.scatter_nd 分散操作（自己起的，不是官方的）

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





