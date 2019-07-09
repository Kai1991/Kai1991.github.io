---
layout:     post
title:      3D点云检测-VoexlNet
subtitle:   基于点云的三维物体检测的端到端学习
date:       2019-06-24
author:     Kai
header-img: img/home-bg-o.jpg
catalog: true
tags:
    - 点云
    - VoexlNet
    - 检测
---

## 前言
使用VoexlNet检测3D点云。VoexlNet模型总结

## VoexlNet网络结构
    * 数据处理层
    * 特征学习提取层
    * 中间卷积层
    * RPN层

## 数据处理层
    * 切割点云数据 获取体素，每个体素内的点云，体素坐标，内个体素内点的个数
    * 抽样数据。目的：减少计算量，TODO
    * 计算体素内每个点到执行的距离，增加特征
    * 数据处理的主要代码

```shell
def process_pointcloud(point_cloud, cls=cfg.DETECT_OBJ):
    # Input:
    #   (N, 4)
    # Output:
    #   voxel_dict
    # print("[+] point_cloud shape: {}".format(point_cloud.shape))  #  [_____, 4]

    # 处理强度字段
	# if there's intensity field
    if point_cloud.shape[1] == 4:
		# intensity field nomarlizing [0,255] to [0,___]
	    point_cloud = np.hstack((point_cloud[:,0:3], (point_cloud[:,3]/400.0).reshape(-1,1)))
    else:
		# fill intensity field to 0
        point_cloud = np.hstack((point_cloud[:,0:3], np.zeros((point_cloud.shape[0], 1))))

    scene_size = np.array([4, 80, 70.4], dtype=np.float32) # 把点云看成这个维度的立方块
    voxel_size = np.array([0.4, 0.2, 0.2], dtype=np.float32) # 把每个分成体素的大小
    grid_size = np.array([10, 400, 352], dtype=np.int64) # 网格大小
    # lidar_coord = np.array([0, 40, 3], dtype=np.float32) # default
    lidar_coord = np.array([0, 40, 3.5], dtype=np.float32)
    max_point_number = 35  # def: 35 每个体素内最大点数

    shifted_coord = point_cloud[:, :3] + lidar_coord #移动点云的位置
    # reverse the point cloud coordinate (X, Y, Z) -> (Z, Y, X)
    voxel_index = np.floor(
        shifted_coord[:, ::-1] / voxel_size).astype(np.int) #每个点的体素索引
    # 构造mask
    bound_x = np.logical_and(
        voxel_index[:, 2] >= 0, voxel_index[:, 2] < grid_size[2])
    bound_y = np.logical_and(
        voxel_index[:, 1] >= 0, voxel_index[:, 1] < grid_size[1])
    bound_z = np.logical_and(
        voxel_index[:, 0] >= 0, voxel_index[:, 0] < grid_size[0])

    bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)
    #获取有效的点云 和点云体素索引
    point_cloud = point_cloud[bound_box] 
    voxel_index = voxel_index[bound_box]

    # [K, 3] coordinate buffer as described in the paper
    coordinate_buffer = np.unique(voxel_index, axis=0) #体素的索引

    K = len(coordinate_buffer)#体素数量
    T = max_point_number # 每个体素网格内最多存储

    # [K, 1] store number of points in each voxel grid
    number_buffer = np.zeros(shape=(K), dtype=np.int64) #每个体素网格内 点云的数量

    # [K, T, 7] feature buffer as described in the paper
    feature_buffer = np.zeros(shape=(K, T, 7), dtype=np.float32)

    # build a reverse index for coordinate buffer
    index_buffer = {}
    for i in range(K):
        index_buffer[tuple(coordinate_buffer[i])] = i

    for voxel, point in zip(voxel_index, point_cloud):
        index = index_buffer[tuple(voxel)]
        number = number_buffer[index]
        if number < T:
            feature_buffer[index, number, :4] = point
            number_buffer[index] += 1

    # 计算每个点到质心的距离 增加 3维特征
    feature_buffer[:, :, -3:] = feature_buffer[:, :, :3] - \
        feature_buffer[:, :, :3].sum(axis=1, keepdims=True)/number_buffer.reshape(K, 1, 1)#增加三列特征

    voxel_dict = {'feature_buffer': feature_buffer,# 特征
                  'coordinate_buffer': coordinate_buffer, # 每个点的体素坐标
                  'number_buffer': number_buffer} # 每个体素里有几个点
    return voxel_dict
```

## 特征学习提取层
    * 逐点云学习特征
    * 每个体素都用128维向量表示
    * VFE结构
<img src="{{ site.baseurl }}/img/2019-6-24-3D_VoexlNet/VFE.png"/>
    * 重点代码

```shell
class VFELayer(object):

    def __init__(self, out_channels, name): #out_channels 32/128
        super(VFELayer, self).__init__()
        self.units = int(out_channels / 2) # 16/64
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            self.dense = tf.layers.Dense(
                self.units, tf.nn.relu, name='dense', _reuse=tf.AUTO_REUSE, _scope=scope) #units 输出的维度大小，改变inputs的最后一维
            self.batch_norm = tf.layers.BatchNormalization(
                name='batch_norm', fused=True, _reuse=tf.AUTO_REUSE, _scope=scope) #批量归一化

    def apply(self, inputs, mask, training):
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        pointwise = self.batch_norm.apply(self.dense.apply(inputs), training)#经过一遍全连接和归一化 shape [K, T, 16/64]

        #n [K, 1, units]
        aggregated = tf.reduce_max(pointwise, axis=1, keep_dims=True) # 获取对大致 

        # [K, T, units]
        repeated = tf.tile(aggregated, [1, cfg.VOXEL_POINT_COUNT, 1])#复制操作 

        # [K, T, 2 * units]
        concatenated = tf.concat([pointwise, repeated], axis=2) # [K, T, 32/128]

        mask = tf.tile(mask, [1, 1, 2 * self.units])

        concatenated = tf.multiply(concatenated, tf.cast(mask, tf.float32)) # 把没有值的点依然给置成0 

        return concatenated
```

## 中间卷积层

    * 逐体素卷积处理

## RPN层

    * RPN结构
 <img src="{{ site.baseurl }}/img/2019-6-24-3D_VoexlNet/volexNet-MiddleNet&RPN.png" />
    * 主要代码 
```shell
class MiddleAndRPN:
    def __init__(self, input, alpha=1.5, beta=1, sigma=3, training=True, name=''):
        # scale = [batchsize, 10, 400/200, 352/240, 128] should be the output of feature learning network
        self.input = input # shape [batchsize, 10, 400/200, 352/240, 128]
        self.training = training
        # groundtruth(target) - each anchor box, represent as △x, △y, △z, △l, △w, △h, rotation
        self.targets = tf.placeholder(
            tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 14]) # 真值
        # postive anchors equal to one and others equal to zero(2 anchors in 1 position)
        self.pos_equal_one = tf.placeholder(
            tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 2])
        self.pos_equal_one_sum = tf.placeholder(tf.float32, [None, 1, 1, 1])
        self.pos_equal_one_for_reg = tf.placeholder(
            tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 14])
        # negative anchors equal to one and others equal to zero
        self.neg_equal_one = tf.placeholder(
            tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 2])
        self.neg_equal_one_sum = tf.placeholder(tf.float32, [None, 1, 1, 1])

        with tf.variable_scope('MiddleAndRPN_' + name):
            # convolutinal middle layers
            temp_conv = ConvMD(3, 128, 64, 3, (2, 1, 1),
                               (1, 1, 1), self.input, name='conv1')# shape [batchsize, 10, 400/200, 352/240, 64]
            temp_conv = ConvMD(3, 64, 64, 3, (1, 1, 1),
                               (0, 1, 1), temp_conv, name='conv2')# shape [batchsize, 10, 400/200, 352/240, 64]
            temp_conv = ConvMD(3, 64, 64, 3, (2, 1, 1),
                               (1, 1, 1), temp_conv, name='conv3')# shape [batchsize, 10, 400/200, 352/240, 64]
            temp_conv = tf.transpose(temp_conv, perm=[0, 2, 3, 4, 1])
            temp_conv = tf.reshape(
                temp_conv, [-1, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128])# shape [batchsize, 400/200, 352/240, 128]

            # rpn
            # block1:
            temp_conv = ConvMD(2, 128, 128, 3, (2, 2), (1, 1),
                               temp_conv, training=self.training, name='conv4')# shape [batchsize, 200/100, 176/120, 128]
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv5')# shape [batchsize, 200/100, 176/120, 128]
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv6')# shape [batchsize, 200/100, 176/120, 128]
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv7')# shape [batchsize, 200/100, 176/120, 128]
            deconv1 = Deconv2D(128, 256, 3, (1, 1), (0, 0),
                               temp_conv, training=self.training, name='deconv1')# shape [batchsize, 200/100, 176/120, 256]

            # block2:
            temp_conv = ConvMD(2, 128, 128, 3, (2, 2), (1, 1),
                               temp_conv, training=self.training, name='conv8')# shape [batchsize, 100/50, 88/60, 128]
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv9')# shape [batchsize, 100/50, 88/60, 128]
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv10')# shape [batchsize, 100/50, 88/60, 128]
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv11')# shape [batchsize, 100/50, 88/60, 128]
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv12')# shape [batchsize, 100/50, 88/60, 128]
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv13')# shape [batchsize, 100/50, 88/60, 128]
            deconv2 = Deconv2D(128, 256, 2, (2, 2), (0, 0),
                               temp_conv, training=self.training, name='deconv2')# shape [batchsize, 200/100, 176/120, 256]

            # block3:
            temp_conv = ConvMD(2, 128, 256, 3, (2, 2), (1, 1),
                               temp_conv, training=self.training, name='conv14')# shape [batchsize, 50/25, 44/30, 256]
            temp_conv = ConvMD(2, 256, 256, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv15')# shape [batchsize, 50/25, 44/30, 256]
            temp_conv = ConvMD(2, 256, 256, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv16')# shape [batchsize, 50/25, 44/30, 256]
            temp_conv = ConvMD(2, 256, 256, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv17')# shape [batchsize, 50/25, 44/30, 256]
            temp_conv = ConvMD(2, 256, 256, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv18')# shape [batchsize, 50/25, 44/30, 256]
            temp_conv = ConvMD(2, 256, 256, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv19')# shape [batchsize, 50/25, 44/30, 256]
            deconv3 = Deconv2D(256, 256, 4, (4, 4), (0, 0),
                               temp_conv, training=self.training, name='deconv3')# shape [batchsize, 200/100, 176/120, 256]

            # final:
            temp_conv = tf.concat([deconv3, deconv2, deconv1], -1)# 3个tensor 拼接
            # Probability score map, scale = [None, 200/100, 176/120, 2]
            p_map = ConvMD(2, 768, 2, 1, (1, 1), (0, 0), temp_conv,
                           training=self.training, activation=False, bn=False, name='conv20') #概率得分
            # Regression(residual) map, scale = [None, 200/100, 176/120, 14]
            r_map = ConvMD(2, 768, 14, 1, (1, 1), (0, 0),
                           temp_conv, training=self.training, activation=False, bn=False, name='conv21')#边框信息
            # softmax output for positive anchor and negative anchor, scale = [None, 200/100, 176/120, 1]
            self.p_pos = tf.sigmoid(p_map)
            #self.p_pos = tf.nn.softmax(p_map, dim=3)
            self.output_shape = [cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH]

            self.cls_pos_loss = (-self.pos_equal_one * tf.log(self.p_pos + small_addon_for_BCE)) / self.pos_equal_one_sum
            self.cls_neg_loss = (-self.neg_equal_one * tf.log(1 - self.p_pos + small_addon_for_BCE)) / self.neg_equal_one_sum
            
            self.cls_loss = tf.reduce_sum( alpha * self.cls_pos_loss + beta * self.cls_neg_loss )
            self.cls_pos_loss_rec = tf.reduce_sum( self.cls_pos_loss )
            self.cls_neg_loss_rec = tf.reduce_sum( self.cls_neg_loss )


            self.reg_loss = smooth_l1(r_map * self.pos_equal_one_for_reg, self.targets *
                                      self.pos_equal_one_for_reg, sigma) / self.pos_equal_one_sum
            self.reg_loss = tf.reduce_sum(self.reg_loss)

            self.loss = tf.reduce_sum(self.cls_loss + self.reg_loss)

            self.delta_output = r_map
            self.prob_output = self.p_pos

```

## Loss
    * 输入的真值 
<img src="{{ site.baseurl }}/img/2019-6-24-3D_VoexlNet/TruthValue.png" />
注 d等于anchor的l和w平方和的根方。
    * Loss
<img src="{{ site.baseurl }}/img/2019-6-24-3D_VoexlNet/Loss.png" /> 



## 实战






