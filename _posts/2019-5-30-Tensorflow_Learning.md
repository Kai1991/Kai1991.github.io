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





