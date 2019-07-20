# !/usr/bin/env python3.7
# -*- encoding: utf-8 -*-
# ***************************************************
# @File    :   model.py
# @Time    :   2019/07/05 20:51:48
# @Author  :   tsing
# @Version :   1.0
# @Contact :   tsingwangfu@163.com
# @License :   (C)Copyright 2019-2020, AKK
# @Desc    :   None
# ***************************************************

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
import matplotlib.pyplot as plt
import scipy.misc as scm
sys.path.append("../..")
import show

class _Bottleneck(tf.keras.Model):
    def __init__(self, filters, block, 
                 downsampling=False, stride=1, **kwargs):
        super(_Bottleneck, self).__init__(**kwargs)

        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + block + '_branch'
        bn_name_base   = 'bn'  + block + '_branch'

        self.downsampling = downsampling
        self.stride = stride
        self.out_channel = filters3
        
        self.conv2a = KL.Conv2D(filters1, (1, 1), strides=(stride, stride),
                                kernel_initializer='he_normal',
                                name=conv_name_base + '2a')
        self.bn2a = KL.BatchNormalization(name=bn_name_base + '2a')

        self.conv2b = KL.Conv2D(filters2, (3, 3), padding='same',
                                kernel_initializer='he_normal',
                                name=conv_name_base + '2b')
        self.bn2b = KL.BatchNormalization(name=bn_name_base + '2b')

        self.conv2c = KL.Conv2D(filters3, (1, 1),
                                kernel_initializer='he_normal',
                                name=conv_name_base + '2c')
        self.bn2c = KL.BatchNormalization(name=bn_name_base + '2c')
         
        if self.downsampling:
            self.conv_shortcut = KL.Conv2D(filters3, (1, 1), strides=(stride, stride),
                                            kernel_initializer='he_normal',
                                            name=conv_name_base + '1')
            self.bn_shortcut = KL.BatchNormalization(name=bn_name_base + '1')     
    
    def call(self, inputs, training=False):
        x = self.conv2a(inputs)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.conv2c(x)
        x = self.bn2c(x, training=training)
        
        if self.downsampling:
            shortcut = self.conv_shortcut(inputs)
            shortcut = self.bn_shortcut(shortcut, training=training)
        else:
            shortcut = inputs
            
        x += shortcut
        x = tf.nn.relu(x)
        
        return x
    
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()

        shape[1] = shape[1] // self.stride
        shape[2] = shape[2] // self.stride
        shape[-1] = self.out_channel
        return tf.TensorShape(shape)        
        

class ResNet(tf.keras.Model):
    def __init__(self, depth, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        if depth not in [50, 101]:
            raise AssertionError('depth must be 50 or 101.')
        self.depth = depth
        self.padding = KL.ZeroPadding2D((3, 3))
        self.conv1 = KL.Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer='he_normal', name='conv1')
        self.bn_conv1 = KL.BatchNormalization(name='bn_conv1')
        self.max_pool = KL.MaxPooling2D((3, 3), strides=(2, 2), padding='same')
        
        self.res2a = _Bottleneck([64, 64, 256], block='2a', downsampling=True, stride=1)
        self.res2b = _Bottleneck([64, 64, 256], block='2b')
        self.res2c = _Bottleneck([64, 64, 256], block='2c')
        
        self.res3a = _Bottleneck([128, 128, 512], block='3a', downsampling=True, stride=2)
        self.res3b = _Bottleneck([128, 128, 512], block='3b')
        self.res3c = _Bottleneck([128, 128, 512], block='3c')
        self.res3d = _Bottleneck([128, 128, 512], block='3d')
        
        self.res4a = _Bottleneck([256, 256, 1024], block='4a', downsampling=True, stride=2)
        self.res4b = _Bottleneck([256, 256, 1024], block='4b')
        self.res4c = _Bottleneck([256, 256, 1024], block='4c')
        self.res4d = _Bottleneck([256, 256, 1024], block='4d')
        self.res4e = _Bottleneck([256, 256, 1024], block='4e')
        self.res4f = _Bottleneck([256, 256, 1024], block='4f')
        if self.depth == 101:
            self.res4g = _Bottleneck([256, 256, 1024], block='4g')
            self.res4h = _Bottleneck([256, 256, 1024], block='4h')
            self.res4i = _Bottleneck([256, 256, 1024], block='4i')
            self.res4j = _Bottleneck([256, 256, 1024], block='4j')
            self.res4k = _Bottleneck([256, 256, 1024], block='4k')
            self.res4l = _Bottleneck([256, 256, 1024], block='4l')
            self.res4m = _Bottleneck([256, 256, 1024], block='4m')
            self.res4n = _Bottleneck([256, 256, 1024], block='4n')
            self.res4o = _Bottleneck([256, 256, 1024], block='4o')
            self.res4p = _Bottleneck([256, 256, 1024], block='4p')
            self.res4q = _Bottleneck([256, 256, 1024], block='4q')
            self.res4r = _Bottleneck([256, 256, 1024], block='4r')
            self.res4s = _Bottleneck([256, 256, 1024], block='4s')
            self.res4t = _Bottleneck([256, 256, 1024], block='4t')
            self.res4u = _Bottleneck([256, 256, 1024], block='4u')
            self.res4v = _Bottleneck([256, 256, 1024], block='4v')
            self.res4w = _Bottleneck([256, 256, 1024], block='4w') 
        
        self.res5a = _Bottleneck([512, 512, 2048], block='5a', downsampling=True, stride=2)
        self.res5b = _Bottleneck([512, 512, 2048], block='5b')
        self.res5c = _Bottleneck([512, 512, 2048], block='5c')
        
        self.out_channel = (256, 512, 1024, 2048)
    
    def call(self, inputs, training=True):
        x = self.padding(inputs)
        x = self.conv1(x)
        x = self.bn_conv1(x, training=training)
        x = tf.nn.relu(x)
        x = self.max_pool(x)
        
        x = self.res2a(x, training=training)
        x = self.res2b(x, training=training)
        C2 = x = self.res2c(x, training=training)
        
        x = self.res3a(x, training=training)
        x = self.res3b(x, training=training)
        x = self.res3c(x, training=training)
        C3 = x = self.res3d(x, training=training)
        
        x = self.res4a(x, training=training)
        x = self.res4b(x, training=training)
        x = self.res4c(x, training=training)
        x = self.res4d(x, training=training)
        x = self.res4e(x, training=training)
        x = self.res4f(x, training=training)
        if self.depth == 101:
            x = self.res4g(x, training=training)
            x = self.res4h(x, training=training)
            x = self.res4i(x, training=training)
            x = self.res4j(x, training=training)
            x = self.res4k(x, training=training)
            x = self.res4l(x, training=training)
            x = self.res4m(x, training=training)
            x = self.res4n(x, training=training)
            x = self.res4o(x, training=training)
            x = self.res4p(x, training=training)
            x = self.res4q(x, training=training)
            x = self.res4r(x, training=training)
            x = self.res4s(x, training=training)
            x = self.res4t(x, training=training)
            x = self.res4u(x, training=training)
            x = self.res4v(x, training=training)
            x = self.res4w(x, training=training) 
        C4 = x
        
        x = self.res5a(x, training=training)
        x = self.res5b(x, training=training)
        C5 = x = self.res5c(x, training=training)
        
        return C2, C3, C4, C5

class FPN(tf.keras.Model):
    def __init__(self, out_channels=256, **kwargs):
        '''Feature Pyramid Networks
        '''
        super(FPN, self).__init__(**kwargs)
        self.out_channels = out_channels
        self.fpn_c2p2 = KL.Conv2D(out_channels, (1, 1), padding='SAME', kernel_initializer='he_normal', name='fpn_c2p2')
        self.fpn_c3p3 = KL.Conv2D(out_channels, (1, 1), padding='SAME', kernel_initializer='he_normal', name='fpn_c3p3')
        self.fpn_c4p4 = KL.Conv2D(out_channels, (1, 1), padding='SAME', kernel_initializer='he_normal', name='fpn_c4p4')
        self.fpn_c5p5 = KL.Conv2D(out_channels, (1, 1), padding='SAME', kernel_initializer='he_normal', name='fpn_c5p5')
        
        self.fpn_p3upsampled = KL.UpSampling2D(size=(2, 2), name='fpn_p3upsampled')
        self.fpn_p4upsampled = KL.UpSampling2D(size=(2, 2), name='fpn_p4upsampled')
        self.fpn_p5upsampled = KL.UpSampling2D(size=(2, 2), name='fpn_p5upsampled')
        
        self.fpn_p2 = KL.Conv2D(out_channels, (3, 3), padding='SAME', kernel_initializer='he_normal', name='fpn_p2')
        self.fpn_p3 = KL.Conv2D(out_channels, (3, 3), padding='SAME', kernel_initializer='he_normal', name='fpn_p3')
        self.fpn_p4 = KL.Conv2D(out_channels, (3, 3), padding='SAME', kernel_initializer='he_normal', name='fpn_p4')
        self.fpn_p5 = KL.Conv2D(out_channels, (3, 3), padding='SAME', kernel_initializer='he_normal', name='fpn_p5')
        
        self.fpn_p6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name='fpn_p6')
        
    def call(self, inputs, training=True):
        C2, C3, C4, C5 = inputs
        
        P5 = self.fpn_c5p5(C5)
        P4 = self.fpn_c4p4(C4) + self.fpn_p5upsampled(P5)
        P3 = self.fpn_c3p3(C3) + self.fpn_p4upsampled(P4)
        P2 = self.fpn_c2p2(C2) + self.fpn_p3upsampled(P3)
        
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = self.fpn_p2(P2)
        P3 = self.fpn_p3(P3)
        P4 = self.fpn_p4(P4)
        P5 = self.fpn_p5(P5)
        
        # subsampling from P5 with stride of 2.
        P6 = self.fpn_p6(P5)
        
        return P2, P3, P4, P5, P6

class AnchorGenerator(object):
    def __init__(self, 
                 scales=(32, 64, 128, 256, 512), 
                 ratios=(0.5, 1, 2), 
                 feature_strides=(4, 8, 16, 32, 64)):
        '''Anchor Generator
        
        Attributes
        ---
            scales: 1D array of anchor sizes in pixels.
            ratios: 1D array of anchor ratios of width/height.
            feature_strides: Stride of the feature map relative to the image in pixels.
        '''
        self.scales = scales
        self.ratios = ratios
        self.feature_strides = feature_strides
     
    def generate_pyramid_anchors(self, img_metas):
        '''Generate the multi-level anchors for Region Proposal Network
        
        Args
        ---
            img_metas: [batch_size, 11]
        
        Returns
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)] in image coordinates.
            valid_flags: [batch_size, num_anchors]
        '''
        # generate anchors
        pad_shape = calc_batch_padded_shape(img_metas)
        
        feature_shapes = [(pad_shape[0] // stride, pad_shape[1] // stride)
                          for stride in self.feature_strides]
        anchors = [
            self._generate_level_anchors(level, feature_shape)
            for level, feature_shape in enumerate(feature_shapes)
        ]
        anchors = tf.concat(anchors, axis=0)

        # generate valid flags
        img_shapes = calc_img_shapes(img_metas)
        valid_flags = [
            self._generate_valid_flags(anchors, img_shapes[i])
            for i in range(img_shapes.shape[0])
        ]
        valid_flags = tf.stack(valid_flags, axis=0)
        
        anchors = tf.stop_gradient(anchors)
        valid_flags = tf.stop_gradient(valid_flags)
        
        return anchors, valid_flags
    
    def _generate_valid_flags(self, anchors, img_shape):
        '''
        Args
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)] in image coordinates.
            img_shape: Tuple. (height, width, channels)
            
        Returns
        ---
            valid_flags: [num_anchors]
        '''
        y_center = (anchors[:, 2] + anchors[:, 0]) / 2
        x_center = (anchors[:, 3] + anchors[:, 1]) / 2
        
        valid_flags = tf.ones(anchors.shape[0], dtype=tf.int32)
        zeros = tf.zeros(anchors.shape[0], dtype=tf.int32)
        
        valid_flags = tf.where(y_center <= img_shape[0], valid_flags, zeros)
        valid_flags = tf.where(x_center <= img_shape[1], valid_flags, zeros)
        
        return valid_flags
    
    def _generate_level_anchors(self, level, feature_shape):
        '''Generate the anchors given the spatial shape of feature map.
        
        Args
        ---
            feature_shape: (height, width)

        Returns
        ---
            numpy.ndarray [anchors_num, (y1, x1, y2, x2)]
        '''
        scale = self.scales[level]
        ratios = self.ratios
        feature_stride = self.feature_strides[level]
        
        # Get all combinations of scales and ratios
        scales, ratios = tf.meshgrid([float(scale)], ratios)
        scales = tf.reshape(scales, [-1])
        ratios = tf.reshape(ratios, [-1])
        
        # Enumerate heights and widths from scales and ratios
        heights = scales / tf.sqrt(ratios)
        widths = scales * tf.sqrt(ratios) 

        # Enumerate shifts in feature space
        shifts_y = tf.multiply(tf.range(feature_shape[0]), feature_stride)
        shifts_x = tf.multiply(tf.range(feature_shape[1]), feature_stride)
        
        shifts_x, shifts_y = tf.cast(shifts_x, tf.float32), tf.cast(shifts_y, tf.float32)
        shifts_x, shifts_y = tf.meshgrid(shifts_x, shifts_y)

        # Enumerate combinations of shifts, widths, and heights
        box_widths, box_centers_x = tf.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = tf.meshgrid(heights, shifts_y)

        # Reshape to get a list of (y, x) and a list of (h, w)
        box_centers = tf.reshape(tf.stack([box_centers_y, box_centers_x], axis=2), (-1, 2))
        box_sizes = tf.reshape(tf.stack([box_heights, box_widths], axis=2), (-1, 2))

        # Convert to corner coordinates (y1, x1, y2, x2)
        boxes = tf.concat([box_centers - 0.5 * box_sizes,
                           box_centers + 0.5 * box_sizes], axis=1)
        return boxes

class AnchorTarget(object):
    def __init__(self,
                 target_means=(0., 0., 0., 0.), 
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 num_rpn_deltas=256,
                 roi_postivate_ratio=0.5,
                 pos_iou_thr=0.7,
                 neg_iou_thr=0.3):
        '''Compute regression and classification targets for anchors.
        
        Attributes
        ---
            target_means: [4]. Bounding box refinement mean for RPN.
            target_stds: [4]. Bounding box refinement standard deviation for RPN.
            num_rpn_deltas: int. Maximal number of Anchors per image to feed to rpn heads.
            roi_postivate_ratio: float.
            pos_iou_thr: float.
            neg_iou_thr: float.
        '''
        self.target_means = target_means
        self.target_stds = target_stds
        self.num_rpn_deltas = num_rpn_deltas
        self.roi_postivate_ratio = roi_postivate_ratio
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.transforms = BoxTransforms()

    def build_targets(self, anchors, valid_flags, gt_boxes, gt_class_ids, gt_imgs):
        '''Given the anchors and GT boxes, compute overlaps and identify positive
        anchors and deltas to refine them to match their corresponding GT boxes.

        Args
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)] in image coordinates.
            valid_flags: [batch_size, num_anchors]
            gt_boxes: [batch_size, num_gt_boxes, (y1, x1, y2, x2)] in image 
                coordinates.
            gt_class_ids: [batch_size, num_gt_boxes] Integer class IDs.
            gt_imgs: just for summary

        Returns
        ---
            rpn_target_matchs: [batch_size, num_anchors] matches between anchors and GT boxes.
                1 = positive anchor, -1 = negative anchor, 0 = neutral anchor
            rpn_target_deltas: [batch_size, num_rpn_deltas, (dy, dx, log(dh), log(dw))] 
                Anchor bbox deltas.
        '''
        rpn_target_matchs = []
        rpn_target_deltas = []
        
        num_imgs = gt_class_ids.shape[0]
        for i in range(num_imgs):
            target_match, target_delta = self._build_single_target(anchors, valid_flags[i], gt_boxes[i], gt_class_ids[i], gt_imgs[i])
            rpn_target_matchs.append(target_match)
            rpn_target_deltas.append(target_delta)
        
        rpn_target_matchs = tf.stack(rpn_target_matchs)
        rpn_target_deltas = tf.stack(rpn_target_deltas)
        
        rpn_target_matchs = tf.stop_gradient(rpn_target_matchs)
        rpn_target_deltas = tf.stop_gradient(rpn_target_deltas)
        
        return rpn_target_matchs, rpn_target_deltas

    def _build_single_target(self, anchors, valid_flags, gt_boxes, gt_class_ids, img):
        '''Compute targets per instance.
        
        Args
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)]
            valid_flags: [num_anchors]
            gt_class_ids: [num_gt_boxes]
            gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]
        
        Returns
        ---
            target_matchs: [num_anchors]
            target_deltas: [num_rpn_deltas, (dy, dx, log(dh), log(dw))] 
        '''
        gt_boxes, _ = trim_zeros(gt_boxes)
        
        target_matchs = tf.zeros(anchors.shape[0], dtype=tf.int32)
        
        # Compute overlaps [num_anchors, num_gt_boxes]
        overlaps = compute_overlaps(anchors, gt_boxes)
        # NOTE for summary--------------------------------------------
        # gt_show = show.visualize_boxes(image=(img.numpy()+np.array([123.7, 116.8, 103.9])).astype(np.uint8), 
        #                           boxes=gt_boxes.numpy().astype(np.int32), 
        #                           labels=1*gt_boxes.shape[0], 
        #                           scores=None, class_labels={0:"BG", 1:"FG"}, 
        #                           groundtruth_box_visualization_color='RoyalBlue',
        #                           agnostic_mode=True, auto_show=True,
        #                           title="gt_boxes")
        # ------------------------------------------------------------
        # Match anchors to GT Boxes
        # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
        # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
        # Neutral anchors are those that don't match the conditions above,
        # and they don't influence the loss function.
        # However, don't keep any GT box unmatched (rare, but happens). Instead,
        # match it to the closest anchor (even if its max IoU is < 0.3).
        
        neg_values = tf.constant([0, -1])
        pos_values = tf.constant([0, 1])
        
        # 1. Set negative anchors first. They get overwritten below if a GT box is
        # matched to them.
        anchor_iou_argmax = tf.argmax(overlaps, axis=1)
        anchor_iou_max = tf.reduce_max(overlaps, reduction_indices=[1])
        print ("anchor_iou_max", anchor_iou_max.numpy()[np.argpartition(anchor_iou_max.numpy(), -5)[-5:]])
        target_matchs = tf.where(anchor_iou_max < self.neg_iou_thr, 
                                 -tf.ones(anchors.shape[0], dtype=tf.int32), 
                                 target_matchs)

        # filter invalid anchors
        target_matchs = tf.where(tf.equal(valid_flags, 1), target_matchs, tf.zeros(anchors.shape[0], dtype=tf.int32))
        # 2. Set anchors with high overlap as positive.
        target_matchs = tf.where(anchor_iou_max >= self.pos_iou_thr, tf.ones(anchors.shape[0], dtype=tf.int32), target_matchs)

        # 3. Set an anchor for each GT box (regardless of IoU value).        
        gt_iou_argmax = tf.argmax(overlaps, axis=0)
        target_matchs = tf.scatter_update(tf.Variable(target_matchs), gt_iou_argmax, 1)
        
        # Subsample to balance positive and negative anchors
        # Don't let positives be more than half the anchors
        ids = tf.where(tf.equal(target_matchs, 1))
        ids = tf.squeeze(ids, 1)
        extra = ids.shape.as_list()[0] - int(self.num_rpn_deltas * self.roi_postivate_ratio)
        if extra > 0:
            # Reset the extra ones to neutral
            ids = tf.random.shuffle(ids)[:extra]
            target_matchs = tf.scatter_update(target_matchs, ids, 0)
        # Same for negative proposals
        ids = tf.where(tf.equal(target_matchs, -1))
        ids = tf.squeeze(ids, 1)
        extra = ids.shape.as_list()[0] - (self.num_rpn_deltas - tf.reduce_sum(tf.cast(tf.equal(target_matchs, 1), tf.int32)))
        if extra > 0:
            # Rest the extra ones to neutral
            ids = tf.random.shuffle(ids)[:extra]
            target_matchs = tf.scatter_update(target_matchs, ids, 0)

        
        # For positive anchors, compute shift and scale needed to transform them
        # to match the corresponding GT boxes.
        ids = tf.where(tf.equal(target_matchs, 1))
        
        a = tf.gather_nd(anchors, ids)
        anchor_idx = tf.gather_nd(anchor_iou_argmax, ids)
        gt = tf.gather(gt_boxes, anchor_idx)
        
        # # NOTE for summary--------------------------------------------
        # anchors_show = show.visualize_boxes(image=(img.numpy()+np.array([123.7, 116.8, 103.9])).astype(np.uint8), 
        #                           boxes=a.numpy().astype(np.int32), 
        #                           labels=1*a.shape[0], 
        #                           scores=None, class_labels={0:"BG", 1:"FG"}, 
        #                           groundtruth_box_visualization_color='RoyalBlue',
        #                           agnostic_mode=True, auto_show=True,
        #                           title="positive_anchors")
        # # ------------------------------------------------------------
        target_deltas = self.transforms.bbox2delta(a, gt, self.target_means, self.target_stds)
        
        padding = tf.maximum(self.num_rpn_deltas - tf.shape(target_deltas)[0], 0)
        target_deltas = tf.pad(target_deltas, [(0, padding), (0, 0)])

        return target_matchs, target_deltas

class BoxTransforms():
    
    @staticmethod
    def bbox2delta(box, gt_box, target_means, target_stds):
        '''Compute refinement needed to transform box to gt_box.
        
        Args
        ---
            box: [..., (y1, x1, y2, x2)]
            gt_box: [..., (y1, x1, y2, x2)]
            target_means: [4]
            target_stds: [4]
        '''
        EPSILON = 1e-10
        target_means = tf.constant(target_means, dtype=tf.float32)
        target_stds = tf.constant(target_stds, dtype=tf.float32)
        box = tf.cast(box, tf.float32)
        gt_box = tf.cast(gt_box, tf.float32)

        height = box[..., 2] - box[..., 0]
        width = box[..., 3] - box[..., 1]
        center_y = box[..., 0] + 0.5 * height
        center_x = box[..., 1] + 0.5 * width

        gt_height = gt_box[..., 2] - gt_box[..., 0]
        gt_width = gt_box[..., 3] - gt_box[..., 1]
        gt_center_y = gt_box[..., 0] + 0.5 * gt_height
        gt_center_x = gt_box[..., 1] + 0.5 * gt_width
        # NOTE avoid nan; maybe you dont need this
        dy = (gt_center_y - center_y) / (height+EPSILON)
        dx = (gt_center_x - center_x) / (width+EPSILON)
        dh = tf.math.log(gt_height / (height+EPSILON))
        dw = tf.math.log(gt_width / (width+EPSILON))

        delta = tf.stack([dy, dx, dh, dw], axis=-1)
        delta = (delta - target_means) / target_stds
        
        return delta                          

    @staticmethod
    def delta2bbox(box, delta, target_means, target_stds):
        '''Compute bounding box based on roi and delta.
        
        Args
        ---
            box: [N, (y1, x1, y2, x2)] box to update
            delta: [N, (dy, dx, log(dh), log(dw))] refinements to apply
            target_means: [4]
            target_stds: [4]
        '''
        target_means = tf.constant(target_means, dtype=tf.float32)
        target_stds = tf.constant(target_stds, dtype=tf.float32)
        delta = delta * target_stds + target_means    
        # Convert to y, x, h, w
        height = box[:, 2] - box[:, 0]
        width = box[:, 3] - box[:, 1]
        center_y = box[:, 0] + 0.5 * height
        center_x = box[:, 1] + 0.5 * width
        
        # Apply delta
        center_y += delta[:, 0] * height
        center_x += delta[:, 1] * width
        height *= tf.exp(delta[:, 2])
        width *= tf.exp(delta[:, 3])
        
        # Convert back to y1, x1, y2, x2
        y1 = center_y - 0.5 * height
        x1 = center_x - 0.5 * width
        y2 = y1 + height
        x2 = x1 + width
        result = tf.stack([y1, x1, y2, x2], axis=1)
        return result

    @staticmethod
    def bbox_clip(box, window):
        '''
        Args
        ---
            box: [N, (y1, x1, y2, x2)]
            window: [4] in the form y1, x1, y2, x2
        '''
        # Split
        wy1, wx1, wy2, wx2 = tf.split(window, 4)
        y1, x1, y2, x2 = tf.split(box, 4, axis=1)
        # Clip
        y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
        x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
        y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
        x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
        clipped = tf.concat([y1, x1, y2, x2], axis=1)
        clipped.set_shape((clipped.shape[0], 4))
        return clipped

    @staticmethod
    def bbox_flip(bboxes, width):
        '''Flip bboxes horizontally.
        
        Args
        ---
            bboxes: [..., 4]
            width: Int or Float
        '''
        y1, x1, y2, x2 = tf.split(bboxes, 4, axis=-1)
        
        new_x1 = width - x2
        new_x2 = width - x1
        
        flipped = tf.concat([y1, new_x1, y2, new_x2], axis=-1)
        
        return flipped


    def bbox_mapping(self, box, img_meta):
        '''
        Args
        ---
            box: [N, 4]
            img_meta: [11]
        '''
        img_meta = parse_image_meta(img_meta)
        scale = img_meta['scale']
        flip = img_meta['flip']
        
        box = box * scale
        if tf.equal(flip, 1):
            box = self.bbox_flip(box, img_meta['img_shape'][1])
        
        return box

    def bbox_mapping_back(self, box, img_meta):
        '''
        Args
        ---
            box: [N, 4]
            img_meta: [11]
        '''
        img_meta = parse_image_meta(img_meta)
        scale = img_meta['scale']
        flip = img_meta['flip']
        if tf.equal(flip, 1):
            box = self.bbox_flip(box, img_meta['img_shape'][1])
        box = box / scale
        return box

def compute_overlaps(boxes1, boxes2, config=None):
    '''Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    '''
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]]) 
    # if config.IOU_TYPE == 'iou':
    #     overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]]) 
    # elif config.IOU_TYPE == 'giou':
    #     # NOTE giou ----------------------------------
    #     y1 = tf.minimum(b1_y1, b2_y1)
    #     x1 = tf.minimum(b1_x1, b2_x1)
    #     y2 = tf.maximum(b1_y2, b2_y2)
    #     x2 = tf.maximum(b1_x2, b2_x2)
    #     con = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    #     giou = iou - (con-union)/con
    #     # --------------------------------------------
    #     overlaps = tf.reshape(giou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])

    return overlaps
        
# ===================================================
def smooth_l1_loss(y_true, y_pred):
    '''Implements Smooth-L1 loss.
    
    Args
    ---
        y_true and y_pred are typically: [N, 4], but could be any shape.
    '''
    # assert y_true is not inf
    diff = tf.abs(y_true - y_pred)
    # print ("y_true", y_true)
    # print ("y_pred", y_pred)
    # print ("diff", diff)
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
    # print ("(1 - less_than_one) * (diff - 0.5)", (1 - less_than_one) * (diff - 0.5))
    # print ("less_than_one * 0.5 * diff**2)", less_than_one * 0.5 * diff**2)
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss

def rpn_class_loss(target_matchs, rpn_class_logits):
    '''RPN anchor classifier loss.
    
    Args
    ---
        target_matchs: [batch_size, num_anchors]. Anchor match type. 1=positive,
            -1=negative, 0=neutral anchor.
        rpn_class_logits: [batch_size, num_anchors, 2]. RPN classifier logits for FG/BG.
    '''

    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = tf.cast(tf.equal(target_matchs, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(tf.not_equal(target_matchs, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Cross entropy loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=anchor_class,
                                                logits=rpn_class_logits)
    
    loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)
    return loss

def rpn_bbox_loss(target_deltas, target_matchs, rpn_deltas):
    '''Return the RPN bounding box loss graph.
    
    Args
    ---
        target_deltas: [batch, num_rpn_deltas, (dy, dx, log(dh), log(dw))].
            Uses 0 padding to fill in unsed bbox deltas.
        target_matchs: [batch, anchors]. Anchor match type. 1=positive,
            -1=negative, 0=neutral anchor.
        rpn_deltas: [batch, anchors, (dy, dx, log(dh), log(dw))]
    '''
    def batch_pack(x, counts, num_rows):
        '''Picks different number of values from each row
        in x depending on the values in counts.
        '''
        outputs = []
        for i in range(num_rows):
            outputs.append(x[i, :counts[i]])
        return tf.concat(outputs, axis=0)
    
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    indices = tf.where(tf.equal(target_matchs, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_deltas = tf.gather_nd(rpn_deltas, indices)

    # Trim target bounding box deltas to the same length as rpn_deltas.
    batch_counts = tf.reduce_sum(tf.cast(tf.equal(target_matchs, 1), tf.int32), axis=1)
    target_deltas = batch_pack(target_deltas, batch_counts,
                            target_deltas.shape.as_list()[0])

    loss = smooth_l1_loss(target_deltas, rpn_deltas)
    
    loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)
    
    return loss

def mrcnn_class_loss(target_matchs, mrcnn_class_logits):
    '''Loss for the classifier head of Faster RCNN.
    
    Args
    ---
        target_matchs:[batch_size * num_rois]. Integer class IDs. Uses zero
            padding to fill in the array.
        mrcnn_class_logits:[batch_size * num_rois, num_classes]
    '''
    
    class_ids = tf.cast(target_matchs, 'int64')
    
    indices = tf.where(tf.not_equal(target_matchs, -1))
    class_ids = tf.gather(class_ids, indices)
    mrcnn_class_logits = tf.gather_nd(mrcnn_class_logits, indices)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=class_ids, logits=mrcnn_class_logits)
    loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)
    return loss

def mrcnn_class2_loss(target_class_ids2, target_class_ids, pred_class_logits2, active_class_ids2):
    """Loss for Mask R-CNN bounding box refinement.
    Params:
    -----------------------------------------------------------
        target_class_ids2:  [batch, num_rois]. Integer class IDs. Uses zero
                            padding to fill in the array.
        pred_class_logits2: [batch, num_rois, num_classes]
        active_class_ids2:  [batch, num_classes]. Has a value of 1 for
                            classes that are in the dataset of the image, and 0
                            for classes that are not in the dataset.
        target_class_ids:   [batch, num_rois]. Integer class IDs.
    """
    # During model building, Keras calls this function with
    # target_class_ids of type float32. Unclear why. Cast it
    # to int to get around it.
    target_class_ids2 = tf.cast(target_class_ids2, 'int64')

    # Find predictions of classes that are not in the dataset.
    pred_class_ids2 = tf.argmax(pred_class_logits2, axis=2)
    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    pred_active = tf.gather(active_class_ids2[0], pred_class_ids2)

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]

    # Gather the deltas (predicted and true) that contribute to loss
    target_class_ids2 = tf.gather(target_class_ids2, positive_roi_ix)
    pred_class_logits2 = tf.gather(pred_class_logits2, positive_roi_ix)
    pred_active = tf.gather(pred_active, positive_roi_ix)

    # if self.config.RCNN_CLASS_LOSS_TYPE == 'cross_entropy':
        # CE_Loss --------------------------------------------------
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids2, logits=pred_class_logits2)
        # ----------------------------------------------------------
    # elif self.config.RCNN_CLASS_LOSS_TYPE == 'focal_loss':
    #     # Focal loss -----------------------------------------------
    #     depth = pred_class_logits2.shape[-1]
    #     loss = focal_loss(prediction_tensor=pred_class_logits2,
    #                     target_tensor=tf.cast(tf.one_hot(target_class_ids2, depth), tf.float32))

    #     loss = tf.reduce_sum(loss, axis=-1)
        # ----------------------------------------------------------
    # Erase losses of predictions of classes that are not in the active
    # classes of the image.

    loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    if tf.reduce_sum(pred_active) > 0:
        loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active),
    else:
        loss = tf.constant(0.0)
    return loss

def mrcnn_bbox_loss(target_deltas, target_matchs, mrcnn_deltas):
    '''Loss for Faster R-CNN bounding box refinement.
    
    Args
    ---
        target_deltas: [batch_size * num_rois, (dy, dx, log(dh), log(dw))]
        target_matchs: [batch_size * num_rois]. Integer class IDs.
        mrcnn_deltas: [batch_size * num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    '''
    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indicies.
    positive_roi_ix = tf.where(target_matchs > 0)[:, 0]
    positive_roi_class_ids = tf.cast(tf.gather(target_matchs, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)
    
    # Gather the deltas (predicted and true) that contribute to loss
    target_deltas = tf.gather(target_deltas, positive_roi_ix)
    mrcnn_deltas = tf.gather_nd(mrcnn_deltas, indices)
    
    # Smooth-L1 Loss
    loss = smooth_l1_loss(target_deltas, mrcnn_deltas)
    
    loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)
    
    return loss

def mrcnn_mask_loss(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.
    Params:
    -----------------------------------------------------------
        target_masks:     [batch*num_rois, height, width].
                        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
        target_class_ids: [batch*num_rois]. Integer class IDs. Zero padded.
        pred_masks:       [batch*proposals, height, width, num_classes] float32 tensor
                        with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.

    # if self.config.MASK_LOSS_TYPE == "binary_ce":
    # NOTE for Compute binary cross entropy ---------------------
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    # Compute binary cross entropy. 
    # If no positive ROIs, then return 0.
    y_pred = tf.gather_nd(pred_masks, indices)
    if tf.size(y_true) > 0:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    else:
        loss = tf.constant(0.0)
        # ----------------------------------------------------------
    # elif self.config.MASK_LOSS_TYPE == "focal_loss":
    #     # Focal Loss -----------------------------------------------
    #     # Only positive ROIs contribute to the loss. And only
    #     # the class specific mask of each ROI.
    #     positive_ix = tf.where(target_class_ids > 0)[:, 0]
    #     # Gather the masks (predicted and true) that contribute to loss
    #     y_true = tf.gather(target_masks, positive_ix)
    #     y_pred = tf.gather(pred_masks, positive_ix)
    #     loss = tf.cond(tf.size(y_true)>0,
    #                     lambda: focal_loss(y_pred, tf.cast(tf.one_hot(tf.cast(y_true, dtype=tf.int32), self.config.NUM_CLASSES), 
    #                             dtype=tf.float32)),
    #                     lambda: tf.constant(0.0))
    
    loss = tf.reduce_mean(loss)

    return loss

class RPNHead(tf.keras.Model):
    def __init__(self, 
                 anchor_scales=(32, 64, 128, 256, 512), 
                 anchor_ratios=(0.5, 1, 2), 
                 anchor_strides=(4, 8, 16, 32, 64),
                 proposal_count=2000, 
                 nms_threshold=0.7, 
                 target_means=(0., 0., 0., 0.), 
                 target_stds=(0.1, 0.1, 0.2, 0.2), 
                 num_rpn_deltas=256,
                 roi_postivate_ratio=0.5,
                 pos_iou_thr=0.7,
                 neg_iou_thr=0.3,
                 **kwags):
        '''Network head of Region Proposal Network.

                                      / - rpn_cls (1x1 conv)
        input - rpn_conv (3x3 conv) -
                                      \ - rpn_reg (1x1 conv)

        Attributes
        ---
            anchor_scales: 1D array of anchor sizes in pixels.
            anchor_ratios: 1D array of anchor ratios of width/height.
            anchor_strides: Stride of the feature map relative 
                to the image in pixels.
            proposal_count: int. RPN proposals kept after non-maximum 
                supression.
            nms_threshold: float. Non-maximum suppression threshold to 
                filter RPN proposals.
            target_means: [4] Bounding box refinement mean.
            target_stds: [4] Bounding box refinement standard deviation.
            num_rpn_deltas: int.
            roi_postivate_ratio: float.
            pos_iou_thr: float.
            neg_iou_thr: float.
        '''
        super(RPNHead, self).__init__(**kwags)
        
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.target_means = target_means
        self.target_stds = target_stds

        self.generator = AnchorGenerator(scales=anchor_scales,ratios=anchor_ratios,feature_strides=anchor_strides)
        
        self.anchor_target = AnchorTarget(target_means=target_means,target_stds=target_stds,
            num_rpn_deltas=num_rpn_deltas,roi_postivate_ratio=roi_postivate_ratio,
            pos_iou_thr=pos_iou_thr,neg_iou_thr=neg_iou_thr)
        self.transforms = BoxTransforms()
        
        # Shared convolutional base of the RPN
        self.rpn_conv_shared = KL.Conv2D(512, (3, 3), padding='same',
                                             kernel_initializer='he_normal', 
                                             name='rpn_conv_shared')
        self.rpn_class_raw = KL.Conv2D(2 * len(anchor_ratios), (1, 1), padding='valid',
                                           kernel_initializer='he_normal', 
                                           name='rpn_class_raw')
        self.rpn_delta_pred = KL.Conv2D(4 * len(anchor_ratios), (1, 1), padding='valid',
                                           kernel_initializer='he_normal', 
                                           name='rpn_bbox_pred')
        
    def call(self, inputs, training=True):
        '''
        Args
        ---
            inputs: [batch_size, feat_map_height, feat_map_width, channels] 
                one level of pyramid feat-maps.
        
        Returns
        ---
            rpn_class_logits: [batch_size, num_anchors, 2]
            rpn_probs: [batch_size, num_anchors, 2]
            rpn_deltas: [batch_size, num_anchors, 4]
        '''
        
        layer_outputs = []
        
        for feat in inputs:
            shared = self.rpn_conv_shared(feat)
            shared = tf.nn.relu(shared)

            x = self.rpn_class_raw(shared)
            rpn_class_logits = tf.reshape(x, [tf.shape(x)[0], -1, 2])
            rpn_probs = tf.nn.softmax(rpn_class_logits)

            x = self.rpn_delta_pred(shared)
            rpn_deltas = tf.reshape(x, [tf.shape(x)[0], -1, 4])
            
            layer_outputs.append([rpn_class_logits, rpn_probs, rpn_deltas])

        outputs = list(zip(*layer_outputs))
        outputs = [tf.concat(list(o), axis=1) for o in outputs]
        rpn_class_logits, rpn_probs, rpn_deltas = outputs
        
        return rpn_class_logits, rpn_probs, rpn_deltas

    def loss(self, rpn_class_logits, rpn_deltas, rpn_target_matchs, rpn_target_deltas):
        '''Calculate rpn loss
        '''

        class_loss = rpn_class_loss(rpn_target_matchs, rpn_class_logits)
        bbox_loss = rpn_bbox_loss(rpn_target_deltas, rpn_target_matchs, rpn_deltas)
        
        return class_loss, bbox_loss
    
    def build_anchor_targets(self, gt_boxes, gt_class_ids, img_metas, 
                           imgs # for summary
                           ):
        anchors, valid_flags = self.generator.generate_pyramid_anchors(img_metas)
        rpn_target_matchs, rpn_target_deltas = self.anchor_target.build_targets(
            anchors, valid_flags, gt_boxes, gt_class_ids, imgs) # imgs just for summary
        return rpn_target_matchs, rpn_target_deltas

    
    def get_proposals(self, 
                      rpn_probs, 
                      rpn_deltas, 
                      img_metas, 
                      imgs, # for summary
                      with_probs=False):
        '''Calculate proposals.
        
        Args
        ---
            rpn_probs: [batch_size, num_anchors, (bg prob, fg prob)]
            rpn_deltas: [batch_size, num_anchors, (dy, dx, log(dh), log(dw))]
            img_metas: [batch_size, 11]
            with_probs: bool.
        
        Returns
        ---
            proposals: [batch_size * num_proposals, (batch_ind, y1, x1, y2, x2))] in 
                normalized coordinates if with_probs is False. 
                Otherwise, the shape of proposals in proposals_list is 
                [batch_size * num_proposals, (batch_ind, y1, x1, y2, x2, probs)]
        
        '''
        anchors, valid_flags = self.generator.generate_pyramid_anchors(img_metas)
        
        rpn_probs = rpn_probs[:, :, 1]
        
        pad_shapes = calc_pad_shapes(img_metas)
        
        proposals_list = [
            self._get_proposals_single(rpn_probs[i], rpn_deltas[i], anchors, valid_flags[i], imgs[i], pad_shapes[i], i, with_probs)
            for i in range(img_metas.shape[0])
        ]
        
        return tf.concat(proposals_list, axis=0, name="proposal_roi")
    
    def _get_proposals_single(self, 
                              rpn_probs, 
                              rpn_deltas, 
                              anchors, 
                              valid_flags, 
                              img, # for summary
                              img_shape,
                              batch_ind,
                              with_probs):
        '''Calculate proposals.
        
        Args
        ---
            rpn_probs: [num_anchors]
            rpn_deltas: [num_anchors, (dy, dx, log(dh), log(dw))]
            anchors: [num_anchors, (y1, x1, y2, x2)] anchors defined in 
                    pixel coordinates.
            valid_flags: [num_anchors]
            img_shape: np.ndarray. [2]. (img_height, img_width)
            batch_ind: int.
            with_probs: bool.
        
        Returns
        ---
            proposals: [num_proposals, (batch_ind, y1, x1, y2, x2)] in normalized 
                coordinates.
        '''
        H, W = img_shape # (IMAGE_SHAPE)
        # filter invalid anchors
        valid_flags = tf.cast(valid_flags, tf.bool)
        
        rpn_probs = tf.boolean_mask(rpn_probs, valid_flags)
        rpn_deltas = tf.boolean_mask(rpn_deltas, valid_flags)
        anchors = tf.boolean_mask(anchors, valid_flags)

        # NOTE for summary--------------------------------------------
        # _pre_nms_limit = min(100, anchors.shape[0])
        # _ix = tf.nn.top_k(rpn_probs, _pre_nms_limit, sorted=True, name="top_anchors100").indices
        
        # _rpn_probs = tf.gather(rpn_probs, _ix)
        # _rpn_deltas = tf.gather(rpn_deltas, _ix)
        # _anchors = tf.gather(anchors, _ix)
        # topanchors_show = show.visualize_boxes(image=(img.numpy()+np.array([123.7, 116.8, 103.9])).astype(np.uint8), 
        #                           boxes=_anchors.numpy().astype(np.int32), 
        #                           labels=1*_anchors.shape[0], 
        #                           scores=_rpn_probs, class_labels={0:"BG", 1:"FG"}, 
        #                           groundtruth_box_visualization_color='RoyalBlue',
        #                           agnostic_mode=True, auto_show=True,
        #                           title="top_anchors100")
        # ------------------------------------------------------------
        # Improve performance
        pre_nms_limit = min(6000, anchors.shape[0])
        ix = tf.nn.top_k(rpn_probs, pre_nms_limit, sorted=True, name="top_anchors").indices
        
        rpn_probs = tf.gather(rpn_probs, ix)
        rpn_deltas = tf.gather(rpn_deltas, ix)
        anchors = tf.gather(anchors, ix)


        # Get refined anchors
        proposals = self.transforms.delta2bbox(anchors, rpn_deltas, self.target_means, self.target_stds)
        
        window = tf.constant([0., 0., H, W], dtype=tf.float32)
        proposals = self.transforms.bbox_clip(proposals, window)
        
        # NOTE for summary--------------------------------------------
        # indices_summary = tf.image.non_max_suppression(proposals, rpn_probs, 100, self.nms_threshold,name="rpn_nms_top100")
        # proposals_summary = tf.gather(proposals, indices_summary)
        # proposals_show = show.visualize_boxes(image=(img.numpy()+np.array([123.7, 116.8, 103.9])).astype(np.uint8), 
        #                           boxes=proposals_summary.numpy().astype(np.int32), 
        #                           labels=1*proposals_summary.shape[0], 
        #                           scores=None, class_labels={0:"BG", 1:"FG"}, 
        #                           groundtruth_box_visualization_color='RoyalBlue',
        #                           agnostic_mode=True, auto_show=True,
        #                           title="proposals_top100")
        # ------------------------------------------------------------
        
        # Normalize
        proposals = proposals / tf.constant([H, W, H, W], dtype=tf.float32)
        
        # NMS
        indices = tf.image.non_max_suppression(
            proposals, rpn_probs, self.proposal_count, self.nms_threshold,
            name="rpn_non_max_suppression")
        proposals = tf.gather(proposals, indices)

        if with_probs:
            proposal_probs = tf.expand_dims(tf.gather(rpn_probs, indices), axis=1)
            proposals = tf.concat([proposals, proposal_probs], axis=1)

        # Pad
        padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
        proposals = tf.pad(proposals, [(0, padding), (0, 0)])
        
        batch_inds = tf.ones((proposals.shape[0], 1)) * batch_ind
        proposals = tf.concat([batch_inds, proposals], axis=1)
        
        return proposals

class DetectionTargetLayer(object):
    def __init__(self,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2), 
                 train_rois_per_image=256,
                 roi_postivate_ratio=0.25,
                 pos_iou_thr=0.5,
                 neg_iou_thr=0.5,
                 config=None):
        '''Compute regression and classification targets for proposals.
        
        Attributes
        ---
            target_means: [4]. Bounding box refinement mean for RCNN.
            target_stds: [4]. Bounding box refinement standard deviation for RCNN.
            train_rois_per_image: int. Maximal number of RoIs per image to feed to bbox heads.

        '''
        self.target_means = target_means
        self.target_stds = target_stds
        self.train_rois_per_image = train_rois_per_image
        self.roi_postivate_ratio = roi_postivate_ratio
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.config = config
        self.transforms = BoxTransforms()
            
    def build_detect_targets(self, proposals, gt_boxes, gt_class_ids, gt_masks, img_metas, imgs):
        '''Generates detection targets for images. Subsamples proposals and
        generates target class IDs, bounding box deltas for each.
        
        Args
        ---
            proposals: [batch_size * num_proposals, (batch_ind, y1, x1, y2, x2)] in normalized coordinates.
                        and crop to pad_shape
            gt_boxes: [batch_size, num_gt_boxes, (y1, x1, y2, x2)] in image coordinates.
            gt_class_ids: [batch_size, num_gt_boxes] Integer class IDs.
            img_metas: [batch_size, 11]
            imgs: just for summary
        Returns
        ---
            rois: [batch_size * num_rois, (batch_ind, y1, x1, y2, x2)] in normalized coordinates
            rcnn_target_matchs: [batch_size * num_rois]. Integer class IDs.
            rcnn_target_deltas: [batch_size * num_rois, (dy, dx, log(dh), log(dw))].
            
        '''
        pad_shapes = calc_pad_shapes(img_metas)
        batch_size = img_metas.shape[0]
        
        proposals = tf.reshape(proposals[:, :5], (batch_size, -1, 5))
        
        rois_list = []
        target_matchs_list = []
        target_deltas_list = []
        target_masks_list = []
        
        for i in range(batch_size):
            rois, target_matchs, target_deltas, target_masks = self._build_single_target(
                proposals[i], gt_boxes[i], gt_class_ids[i], gt_masks[i], pad_shapes[i], i, imgs[i])
            rois_list.append(rois)
            target_matchs_list.append(target_matchs)
            target_deltas_list.append(target_deltas)
            target_masks_list.append(target_masks)

        rois = tf.concat(rois_list, axis=0)
        target_matchs = tf.concat(target_matchs_list, axis=0)
        target_deltas = tf.concat(target_deltas_list, axis=0)
        target_masks = tf.concat(target_masks_list, axis=0)

        rois = tf.stop_gradient(rois)
        target_matchs = tf.stop_gradient(target_matchs)
        target_deltas = tf.stop_gradient(target_deltas)
        target_masks = tf.stop_gradient(target_masks)
        
        return rois, target_matchs, target_deltas, target_masks
    
    def _build_single_target(self, proposals, gt_boxes, gt_class_ids, gt_masks, img_shape, batch_ind, img):
        '''
        Args
        ---
            proposals: [num_proposals, (batch_ind, y1, x1, y2, x2)] in normalized coordinates.
            gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]
            gt_class_ids: [num_gt_boxes]
            gt_masks: [img_height, img_width, num_gt_boxes]
            img_shape: np.ndarray. [2]. (img_height, img_width)
            batch_ind: int.
            imgs: just for summary
            
        Returns
        ---
            rois: [num_rois, (batch_ind, y1, x1, y2, x2)]
            target_matchs: [num_rois]
            target_deltas: [num_rois, (dy, dx, log(dh), log(dw))]
            target_masks: [mini_h, mini_w, num_rois]
        '''
        H, W = img_shape
        
        trimmed_proposals, _ = trim_zeros(proposals[:, 1:], "trim_proposals")
        gt_boxes, non_zeros = trim_zeros(gt_boxes, "trim_gt_boxes")
        gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros, "trim_gt_class_ids")
        # norm gt_boxes
        gt_boxes = gt_boxes / tf.constant([H, W, H, W], dtype=tf.float32)
        
        # NOTE mrcnn
        gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2, name="trim_gt_masks")
        
        """
        crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
        non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
        crowd_boxes = tf.gather(gt_boxes, crowd_ix)
        gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
        gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
        gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)
        # # ---------------------------NOTE for label2,read and rbox
        # gt_class_ids2 = tf.gather(gt_class_ids2, non_crowd_ix)
        # gt_text_embed = tf.gather(gt_text_embed, non_crowd_ix)
        # gt_embed_length = tf.gather(gt_embed_length, non_crowd_ix)
        # gt_rboxes = tf.gather(gt_rboxes, non_crowd_ix)
        overlaps = compute_overlaps(trimmed_proposals, gt_boxes, config=None)
        crowd_overlaps = compute_overlaps(trimmed_proposals, crowd_boxes, config=None)
        crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
        # roi_iou_max = tf.reduce_max(overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
        # Determine positive and negative ROIs
        roi_iou_max = tf.reduce_max(overlaps, axis=1)
        # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
        positive_roi_bool = (roi_iou_max >= self.pos_iou_thr)
        positive_indices = tf.where(positive_roi_bool)[:, 0]
        negative_indices = tf.where(tf.math.logical_and(roi_iou_max < self.neg_iou_thr, no_crowd_bool))[:, 0]
        """
        overlaps = compute_overlaps(trimmed_proposals, gt_boxes, config=None)
        anchor_iou_argmax = tf.argmax(overlaps, axis=1)
        roi_iou_max = tf.reduce_max(overlaps, axis=1)

        positive_roi_bool = (roi_iou_max >= self.pos_iou_thr)
        positive_indices = tf.where(positive_roi_bool)[:, 0]
        negative_indices = tf.where(roi_iou_max < self.neg_iou_thr)[:, 0]
        # Subsample ROIs. Aim for 33% positive
        # Positive ROIs
        positive_count = int(self.train_rois_per_image * self.roi_postivate_ratio)
        # int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
        positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
        positive_count = tf.shape(positive_indices)[0]
        
        # Negative ROIs. Add enough to maintain positive:negative ratio.
        r = 1.0 / self.roi_postivate_ratio
        negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
        negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
        
        # Gather selected ROIs
        positive_rois = tf.gather(proposals, positive_indices)
        negative_rois = tf.gather(proposals, negative_indices)

        # Assign positive ROIs to GT boxes.
        positive_overlaps = tf.gather(overlaps, positive_indices)
        target_box_assignment = tf.argmax(positive_overlaps, axis=1)
        target_boxes = tf.gather(gt_boxes, target_box_assignment)
        target_class_ids = tf.gather(gt_class_ids, target_box_assignment)

        target_deltas = self.transforms.bbox2delta(positive_rois[:, 1:], target_boxes, self.target_means, self.target_stds)
        # print ("target_deltas", target_deltas)
        #-NOTE-------------------------------------------------------------
        # Assign positive ROIs to GT masks
        # Permute masks to [N, height, width, 1]
        transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
        # Pick the right mask for each ROI
        target_masks = tf.gather(transposed_masks, target_box_assignment)

        # Compute mask targets
        # print (positive_rois)
        boxes = positive_rois[:, 1:]
        if self.config.USE_MINI_MASK:
            # Transform ROI coordinates from normalized image space
            # to normalized mini-mask space.
            y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(target_boxes, 4, axis=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = tf.concat([y1, x1, y2, x2], 1)
        box_ids = tf.range(0, tf.shape(target_masks)[0])
        target_masks = tf.image.crop_and_resize(tf.cast(target_masks, tf.float32), boxes,
                                                box_ids, self.config.MASK_SHAPE)
        # Remove the extra dimension from masks.
        # Shape is [num_rois, 28, 28]
        target_masks = tf.squeeze(target_masks, axis=3)

        # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
        # binary cross entropy loss.
        if not self.config.SOFT_MASK:
            target_masks = tf.round(target_masks)
        #------------------------------------------------------------------

        rois = tf.concat([positive_rois, negative_rois], axis=0)
        
        N = tf.shape(negative_rois)[0]
        P = tf.maximum(self.train_rois_per_image - tf.shape(rois)[0], 0)
        
        rois = tf.pad(rois, [(0, P), (0, 0)])
        
        target_class_ids = tf.pad(target_class_ids, [(0, N + P)])
        target_deltas = tf.pad(target_deltas, [(0, N + P), (0, 0)])

        # NOTE mrcnn
        target_masks = tf.pad(target_masks, [(0, N + P), (0, 0), (0, 0)])

        return rois, target_class_ids, target_deltas, target_masks

class PyramidROIAlign(tf.keras.layers.Layer):
    def __init__(self, pool_shape, config=None, roi_for_read=False, **kwargs):
        '''Implements ROI Pooling on multiple levels of the feature pyramid.

        Attributes
        ---
            pool_shape: (height, width) of the output pooled regions.
                Example: (7, 7)
        '''
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)
        self.config = config

    def call(self, inputs, training=True):
        '''
        Args
        ---
            rois: [batch_size * num_rois, (batch_ind, y1, x1, y2, x2)] in normalized coordinates.
            feature_map_list: List of [batch_size, height, width, channels].
                feature maps from different levels of the pyramid.
            img_metas: [batch_size, 11]

        Returns
        ---
            pooled_rois_list: list of [batch_size * num_rois, pooled_height, pooled_width, channels].
                The width and height are those specific in the pool_shape in the layer
                constructor.
        '''
        rois, feature_map_list, img_metas = inputs

        pad_shapes = calc_pad_shapes(img_metas)
        
        pad_areas = pad_shapes[:, 0] * pad_shapes[:, 1]
        
        roi_indices = tf.cast(rois[:, 0], tf.int32)
        rois = rois[:, 1:]
        
        areas = tf.cast(pad_areas[0], tf.float32)

        
        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(rois, 4, axis=1)
        h = y2 - y1
        w = x2 - x1
        
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4

        roi_level = tf.math.log(tf.sqrt(tf.squeeze(h * w, 1)) / tf.cast((224.0 / tf.sqrt(areas * 1.0)), tf.float32)) / tf.math.log(2.0)
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))


        
        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled_rois = []
        roi_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_rois = tf.gather_nd(rois, ix)

            # ROI indicies for crop_and_resize.
            level_roi_indices = tf.gather_nd(roi_indices, ix)

            # Keep track of which roi is mapped to which level
            roi_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_rois = tf.stop_gradient(level_rois)
            level_roi_indices = tf.stop_gradient(level_roi_indices)
            

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_rois, pool_height, pool_width, channels]
            pooled_rois.append(tf.image.crop_and_resize(
                feature_map_list[i], level_rois, level_roi_indices, self.pool_shape,
                method="bilinear"))
            
        # Pack pooled features into one tensor
        pooled_rois = tf.concat(pooled_rois, axis=0)

        # Pack roi_to_level mapping into one array and add another
        # column representing the order of pooled rois
        roi_to_level = tf.concat(roi_to_level, axis=0)
        roi_range = tf.expand_dims(tf.range(tf.shape(roi_to_level)[0]), 1)
        roi_to_level = tf.concat([tf.cast(roi_to_level, tf.int32), roi_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original rois
        # Sort roi_to_level by batch then roi index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = roi_to_level[:, 0] * 100000 + roi_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(roi_to_level)[0]).indices[::-1]
        ix = tf.gather(roi_to_level[:, 1], ix)
        pooled_rois = tf.gather(pooled_rois, ix)
        
        return pooled_rois

class Classifier(tf.keras.Model):
    def __init__(self, num_classes, 
                 pool_size=(7, 7),
                 target_means=(0., 0., 0., 0.), 
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 min_confidence=0.7,
                 nms_threshold=0.3,
                 max_instances=100,
                 **kwags):
        super(Classifier, self).__init__(**kwags)
        self.num_classes = num_classes
        self.pool_size = tuple(pool_size)
        self.target_means = target_means
        self.target_stds = target_stds
        self.min_confidence = min_confidence
        self.nms_threshold = nms_threshold
        self.max_instances = max_instances
        self.transforms = BoxTransforms()
        self.roialign = PyramidROIAlign(self.pool_size, None, name="roi_align_classifier")
        self.rcnn_class_conv1 = KL.Conv2D(1024, self.pool_size, padding='valid', name='rcnn_class_conv1')
        self.rcnn_class_bn1 = KL.BatchNormalization(name='rcnn_class_bn1')
        self.rcnn_class_conv2 = KL.Conv2D(1024, (1, 1), name='rcnn_class_conv2')
        self.rcnn_class_bn2 = KL.BatchNormalization(name='rcnn_class_bn2')
        self.rcnn_class_logits = KL.Dense(num_classes, name='rcnn_class_logits')
        self.rcnn_delta_fc = KL.Dense(num_classes * 4, name='rcnn_bbox_fc')

    def call(self, inputs, training=True):
        '''
        Args
        ---
            pooled_rois: [batch_size * num_rois, pool_size, pool_size, channels]
        
        Returns
        ---
            rcnn_class_logits: [batch_size * num_rois, num_classes]
            rcnn_probs: [batch_size * num_rois, num_classes]
            rcnn_deltas: [batch_size * num_rois, num_classes, (dy, dx, log(dh), log(dw))]
        '''
        rois, feature_maps, img_metas = inputs
        pooled_rois = self.roialign(inputs, training=training)
        x = self.rcnn_class_conv1(pooled_rois)
        x = self.rcnn_class_bn1(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.rcnn_class_conv2(x)
        x = self.rcnn_class_bn2(x, training=training)
        x = tf.nn.relu(x)
        
        x = tf.squeeze(tf.squeeze(x, 2), 1)
        logits = self.rcnn_class_logits(x)
        probs = tf.nn.softmax(logits)
        
        deltas = self.rcnn_delta_fc(x)
        deltas = tf.reshape(deltas, (-1, self.num_classes, 4))
        
        return logits, probs, deltas

    def loss(self, rcnn_class_logits, rcnn_deltas, 
             rcnn_target_matchs, rcnn_target_deltas):
        '''Calculate RCNN loss
        '''
        rcnn_class_loss = mrcnn_class_loss(rcnn_target_matchs, rcnn_class_logits)
        rcnn_bbox_loss = mrcnn_bbox_loss(rcnn_target_deltas, rcnn_target_matchs, rcnn_deltas)
        
        return rcnn_class_loss, rcnn_bbox_loss
        
    def get_bboxes(self, rcnn_probs, rcnn_deltas, rois, img_metas):
        '''
        Args
        ---
            rcnn_probs: [batch_size * num_rois, num_classes]
            rcnn_deltas: [batch_size * num_rois, num_classes, (dy, dx, log(dh), log(dw))]
            rois: [batch_size * num_rois, (batch_ind, y1, x1, y2, x2)]
            img_meta_list: [batch_size, 11]
        
        Returns
        ---
            detections_list: List of [num_detections, (y1, x1, y2, x2, class_id, score)]
                coordinates are in pixel coordinates.
        '''
        batch_size = img_metas.shape[0]
        rcnn_probs = tf.reshape(rcnn_probs, (batch_size, -1, self.num_classes))
        rcnn_deltas = tf.reshape(rcnn_deltas, (batch_size, -1, self.num_classes, 4))
        rois = tf.reshape(rois, (batch_size, -1, 5))[:, :, 1:5]
        
        pad_shapes = calc_pad_shapes(img_metas)
        # img_shapes = calc_img_shapes(img_metas) # NOTE
        
        detections_list = [
            self._get_bboxes_single(rcnn_probs[i], rcnn_deltas[i], rois[i], pad_shapes[i]) #pad_shapes[i])
            for i in range(img_metas.shape[0])
        ]
        return detections_list
    
    def _get_bboxes_single(self, rcnn_probs, rcnn_deltas, rois, img_shape):
        '''
        Args
        ---
            rcnn_probs: [num_rois, num_classes]
            rcnn_deltas: [num_rois, num_classes, (dy, dx, log(dh), log(dw))]
            rois: [num_rois, (y1, x1, y2, x2)]
            img_shape: np.ndarray. [2]. (img_height, img_width)       
        '''
        H, W = img_shape   
        # Class IDs per ROI
        class_ids = tf.argmax(rcnn_probs, axis=1, output_type=tf.int32)
        # Class probability of the top class of each ROI
        indices = tf.stack([tf.range(rcnn_probs.shape[0]), class_ids], axis=1)
        class_scores = tf.gather_nd(rcnn_probs, indices)
        # Class-specific bounding box deltas
        deltas_specific = tf.gather_nd(rcnn_deltas, indices)
        # Apply bounding box deltas
        # Shape: [num_rois, (y1, x1, y2, x2)] in normalized coordinates        
        refined_rois = self.transforms.delta2bbox(rois, deltas_specific, self.target_means, self.target_stds)
        
        # Clip boxes to image window
        refined_rois *= tf.constant([H, W, H, W], dtype=tf.float32)
        window = tf.constant([0., 0., H * 1., W * 1.], dtype=tf.float32)
        refined_rois = self.transforms.bbox_clip(refined_rois, window)
        
        
        # Filter out background boxes
        keep = tf.where(class_ids > 0)[:, 0]
        # Filter out low confidence boxes
        if self.min_confidence:
            conf_keep = tf.where(class_scores >= self.min_confidence)[:, 0]
            keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                            tf.expand_dims(conf_keep, 0))
            keep = tf.sparse.to_dense(keep)[0]
            
        # Apply per-class NMS
        # 1. Prepare variables
        pre_nms_class_ids = tf.gather(class_ids, keep)
        pre_nms_scores = tf.gather(class_scores, keep)
        pre_nms_rois = tf.gather(refined_rois,   keep)
        unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

        def nms_keep_map(class_id):
            '''Apply Non-Maximum Suppression on ROIs of the given class.'''
            # Indices of ROIs of the given class
            ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
            # Apply NMS
            class_keep = tf.image.non_max_suppression(
                            tf.gather(pre_nms_rois, ixs),
                            tf.gather(pre_nms_scores, ixs),
                            max_output_size=self.max_instances,
                            iou_threshold=self.nms_threshold)
            # Map indices
            class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
            return class_keep

        # 2. Map over class IDs
        nms_keep = []
        for i in range(unique_pre_nms_class_ids.shape[0]):
            nms_keep.append(nms_keep_map(unique_pre_nms_class_ids[i]))
        nms_keep = tf.concat(nms_keep, axis=0)
        
        # 3. Compute intersection between keep and nms_keep
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0), tf.expand_dims(nms_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]
        # Keep top detections
        roi_count = self.max_instances
        class_scores_keep = tf.gather(class_scores, keep)
        num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
        top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
        keep = tf.gather(keep, top_ids)  
        
        # bboxes = tf.gather(refined_rois, keep),
        # class_ids = tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        # class_scores = tf.gather(class_scores, keep)[..., tf.newaxis]
        detections = tf.concat([
            tf.gather(refined_rois, keep),
            tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
            tf.gather(class_scores, keep)[..., tf.newaxis]
            ], axis=1)
        
        return detections

class Masker(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(Masker, self).__init__(**kwargs)
        self.config = config
        self.roialign = PyramidROIAlign(config.MASK_POOL_SIZE, None, name="roi_align_classifier")
        for i in range(1, 5):
            setattr(self, f"mrcnn_mask_conv{i}", KL.Conv2D(256, (3, 3), padding="same", name=f"mrcnn_mask_conv{i}"))
            setattr(self, f"mrcnn_mask_bn{i}", KL.BatchNormalization(name=f"mrcnn_mask_bn{i}"))
        self.tdeconv = KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu", name="mrcnn_mask_deconv")
        # if config.MODEL == "mrcnn":
        self.tmask = KL.Conv2D(config.NUM_CLASSES, (1, 1), strides=1, activation="sigmoid", name="mrcnn_mask")
        # else:
        #     self.tmask = KL.Conv2D(1, (1, 1), strides=1, activation="sigmoid", name="mrcnn_mask")
        #     self.concat = KL.Concatenate(axis=-1)
        self.transforms = BoxTransforms()
    def call(self, inputs, training=True):
        '''
        Args
        ---
            pooled_rois: [batch_size * num_rois, pool_size, pool_size, channels]
        
        Returns
        ---
            rcnn_class_logits: [batch_size * num_rois, num_classes]
            rcnn_probs: [batch_size * num_rois, num_classes]
            rcnn_deltas: [batch_size * num_rois, num_classes, (dy, dx, log(dh), log(dw))]
        '''
        rois, feature_maps, img_metas = inputs
        x = self.roialign(inputs, training=training)
        # Conv layers
        for i in range(1, 5):
            x = getattr(self, f"mrcnn_mask_conv{i}")(x)
            x = getattr(self, f"mrcnn_mask_bn{i}")(x, training=training)
            x = tf.nn.relu(x)
        x = self.tdeconv(x)
        x = self.tmask(x)
        if self.config.MODEL == "smrcnn":
            x = self.concat([x for i in range(2)])
        return x
    
    def loss(self, target_masks, target_class_ids, pred_masks):
        return mrcnn_mask_loss(target_masks, target_class_ids, pred_masks)

    @staticmethod
    def unmold_mask(mask, bbox, image_shape, softmask=False):
        """Converts a mask generated by the neural network to a format similar
        to its original shape.
        Params:
        -----------------------------------------------------------
            mask: [height, width] of type float. A small, typically 28x28 mask.
            bbox: [y1, x1, y2, x2]. The box to fit the mask in.
            image_shape: 
        Returns:
        ----------------------------------------------------------- 
            a binary mask with the same size as the original image.
        """
        mask_type = tf.float32 if softmask else tf.bool
        threshold = 0.5
        y1, x1, y2, x2 = tf.cast(bbox, tf.int32)
        mask = tf.image.resize(tf.expand_dims(mask, axis=-1), (y2 - y1, x2 - x1))
        mask = tf.squeeze(mask, axis=-1)
        ones = tf.ones_like(mask)
        zeros = tf.ones_like(mask)
        if not softmask:
            mask = tf.cast(tf.where(mask >= threshold, ones, zeros), tf.bool)
        # Put the mask in the right location.
        # plt.imshow(mask.numpy())
        # plt.show()
        full_mask = tf.pad(mask, [[y1, image_shape[0]-y2], [x1, image_shape[1]-x2]])
        # full_mask = tf.zeros(image_shape[:2], dtype=mask_type)
        # full_mask.numpy()[y1:y2, x1:x2] = mask.numpy()
        return full_mask


class Recognition(tf.keras.Model):
    def __init__(self, config, name='mrcnn_recognition_model', **kwargs):
        super(Recognition, self).__init__(name=name, **kwargs)
        self.config = config
        self.roialign = PyramidROIAlign([4, 13], config, True, name="roi_align_recognition")
        for i in range(1, 5):
            setattr(self, f"mrcnn_recognition_conv{i}", KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name=f"mrcnn_recognition_conv{i}"))
            setattr(self, f"mrcnn_recognition_{config.NORM_TYPE}{i}", KL.TimeDistributed(NORMALIZATION[config.NORM_TYPE](), name=f"mrcnn_recognition_{config.NORM_TYPE}{i}"))
            setattr(self, f"relu{i}", KL.Activation('relu'))
        self.permute = KL.Permute((1,3,2,4), name='mrcnn_recognition_permute')
        self.flatten = KL.TimeDistributed(KL.Flatten(), name='mrcnn_recognition_flatten')
        self.fc = KL.Dense(config.NUM_CHARACTER, name='out', activation='softmax')

    def call(self, inputs, training=False):
        rois, image_meta, mask, feature_maps = inputs
        x = self.roialign([rois, image_meta] + feature_maps)
        # Conv layers
        for i in range(1, 5):
            x = getattr(self, f"mrcnn_recognition_conv{i}")(x)
            x = getattr(self, f"mrcnn_recognition_{self.config.NORM_TYPE}{i}")(x, training=training)
            x = getattr(self, f"relu{i}")(x)
        x = self.permute(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class L1_distance(tf.keras.Model):
    def __init__(self, feature_maps=128, name='Tx', **kwargs):
        super(L1_distance, self).__init__(name=name, **kwargs)
        self.feature_maps = feature_maps
        self.gap = KL.GlobalAveragePooling2D()
        self.concat = KL.Concatenate()
        self.conv = KL.Conv2D(feature_maps, (1,1), name=f"fpn_distance_{name}")
    def call(self, inputs):
        P,T = inputs
        T = self.gap(T)
        T = tf.expand_dims(tf.expand_dims(T, axis=1), axis=1)
        L1 = tf.math.subtract(P, T)
        L1 = tf.math.abs(L1)
        D = self.concat([P, L1])
        if self.feature_maps:
            D = self.conv(D)
        return D


# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = "model"  # Override in sub-classes
    TIME = 20190627
    MODEL_DIR = f"logs/{NAME}{TIME}"
    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # Backbone network architecture
    # Supported values are: resnet50, resnet101, mobilenetv224
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet101"
    NECK = "fpn"
    # mrcnn or frcnn
    MODEL = 'mrcnn'

    # Size of the fully-connected layers in the classification graph
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    # Size of the top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE = FPN_FEATUREMAPS = 256

    # Number of classification classes (including background)
    NUM_CLASSES = 1  # Override in sub-classes
    NUM_CLASSES2 = 1
    # Number of targets for few-shot learning
    NUM_TARGETS = 1

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_POS_IOU_THR = 0.7 # anchor match
    RPN_NEG_IOU_THR = 0.3 # anchor not match
    RPN_NMS_THRESHOLD = 0.7
    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    
    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    PRE_NMS_LIMIT = 6000

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resizing
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024

    # NOTE for siamese
    # Input target resizing
    # The same rukes apply as for image resizing
    # Currently the resizing is done through TARGET_PADDING through which the 
    # target will automatically be padded to [max_dim, max_dim]
    TARGET_PADDING = True
    TARGET_MAX_DIM = 96
    TARGET_MIN_DIM = 75

    # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    # up scaling. For example, if set to 2 then images are scaled up to double
    # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    # However, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    IMAGE_MIN_SCALE = 0
    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT = 3

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    STD_PIXEL = np.array([1., 1., 1.])
    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 256

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = (7, 7)
    MASK_POOL_SIZE = (14, 14)

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_MEANS = np.array([0., 0., 0., 0.])
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    RPN_RBBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2, 0.05])
    MRCNN_BBOX_MEANS = np.array([0., 0., 0., 0.])
    MRCNN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    MRCNN_RBBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2, 0.05])

    DETECTION_POS_IOU_THR = 0.5
    DETECTION_NEG_IOU_THR = 0.5

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Use AdamW optimizer
    ADAMW = False 
    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.0001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # ************************* NOTE for 2 label dataset 
    HAVE_LABEL2 = False
    # rpn class loss_type: cross_entropy or focalloss
    RPN_CLASS_LOSS_TYPE = 'cross_entropy'
    # rpn bbox loss type: smooth_l1_loss or shape_gradient_smooth_l1_loss
    RPN_BBOX_LOSS_TYPE = 'smooth_l1_loss'
    # rcnn class loss type: cross entropy or focalloss
    RCNN_CLASS_LOSS_TYPE = 'cross_entropy'
    # rcnn bbox loss type: smooth_l1_loss or shape_gradient_smooth_l1_loss
    RCNN_BBOX_LOSS_TYPE = 'smooth_l1_loss'
    # mask loss type: binary_ce or focal_loss or lovasz_loss
    MASK_LOSS_TYPE = 'binary_ce'
    
    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "mrcnn_class_loss2": 1,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    IOU_TYPE = 'iou'

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True
    NO_AUGMENT_SOURCES = []

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False  # Defaulting to False since batch size is often small
    NORM_TYPE = 'bn' # 'bn', 'gn', 'sn'

    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0

    # add text_recognition or not
    READ = False
    READ_IMG_HEIGHT = 32
    READ_IMG_HEIGHT = 32
    READ_IMG_WIDTH = 280
    MAX_LABEL_LENGTH = 30


    # use soft mask or not
    SOFT_MASK = False
    PAN = False
    NAS_FPN = False
    TWO_HEAD = False
    RBOX = False
    GLOBAL_MASK = False
    MASKIOU = False
    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM,
                self.IMAGE_CHANNEL_COUNT])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM,
                self.IMAGE_CHANNEL_COUNT])
        self.TARGET_SHAPE = np.array([self.TARGET_MAX_DIM, self.TARGET_MAX_DIM, 3])
        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES + self.NUM_CLASSES2  


    def display(self, save_config=False, save_path='config.yml'):
        """Display Configuration values."""
        print(f"Configurations: \n{'----'*30}")
        if save_config:
            with open(save_path, 'w', encoding='utf-8') as yaml_file:
                yaml_obj = {}
                for a in dir(self):
                    if not a.startswith("__") and not callable(getattr(self, a)):
                        yaml_obj[a] = getattr(self, a)
                        print(f"{a:30} {getattr(self, a)}")
                yaml.dump(yaml_obj, yaml_file)
        else:
            for a in dir(self):
                if not a.startswith("__") and not callable(getattr(self, a)):
                    print(f"{a:30} {getattr(self, a)}")


class Model(tf.keras.Model):
    def __init__(self, config, **kwags):
        super(Model, self).__init__(name=config.NAME, **kwags)
        self.config = config 
        self.backbone = ResNet(depth=101, name='res_net')
        self.fpn = FPN(name='fpn')
        self.rpn = RPNHead(
            anchor_scales=config.RPN_ANCHOR_SCALES,
            anchor_ratios=config.RPN_ANCHOR_RATIOS,
            anchor_strides=config.BACKBONE_STRIDES,
            proposal_count=config.POST_NMS_ROIS_TRAINING,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            target_means=config.RPN_BBOX_MEANS,
            target_stds=config.RPN_BBOX_STD_DEV,
            num_rpn_deltas=config.RPN_TRAIN_ANCHORS_PER_IMAGE,
            roi_postivate_ratio=config.ROI_POSITIVE_RATIO,
            pos_iou_thr=config.RPN_POS_IOU_THR,
            neg_iou_thr=config.RPN_NEG_IOU_THR,
            name='rpn_head')
        # Target Generator for the second stage.
        self.detect_target = DetectionTargetLayer(
            target_means=config.MRCNN_BBOX_MEANS,
            target_stds=config.RPN_BBOX_STD_DEV, 
            train_rois_per_image=config.TRAIN_ROIS_PER_IMAGE,
            roi_postivate_ratio=config.ROI_POSITIVE_RATIO,
            pos_iou_thr=config.DETECTION_POS_IOU_THR,
            neg_iou_thr=config.DETECTION_NEG_IOU_THR,
            config=config)
        self.classifier_and_regression = Classifier(
            num_classes=config.NUM_CLASSES,
            pool_size=config.POOL_SIZE,
            target_means=config.MRCNN_BBOX_MEANS,
            target_stds=config.MRCNN_BBOX_STD_DEV,
            min_confidence=config.DETECTION_MIN_CONFIDENCE,
            nms_threshold=config.DETECTION_NMS_THRESHOLD,
            max_instances=config.DETECTION_MAX_INSTANCES,
            name='classifier')
        self.masker = Masker(config=config, name='masker')
    def call(self, inputs, training=True):      
        if training: # training
            imgs, img_metas, gt_boxes, gt_class_ids, gt_masks, gt_global_mask = inputs
        else: # inference
            imgs, img_metas = inputs
        
        C2, C3, C4, C5 = self.backbone(imgs, training=training)
        P2, P3, P4, P5, P6 = self.fpn([C2, C3, C4, C5], training=training)
        
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        rpn_class_logits, rpn_probs, rpn_deltas = self.rpn(rpn_feature_maps, training=training)

        # NOTE imgs just for summary
        proposals = self.rpn.get_proposals(rpn_probs, rpn_deltas, img_metas, imgs)
        # print ("proposals", proposals)
        # proposals_show = tf.image.draw_bounding_boxes(imgs, tf.reshape(proposals, [2, -1, 5])[:, 1:], name=None)
        # plt.imshow(proposals_show[0].numpy())
        # plt.show()
        
        if training:
            # for loss,arg imgs for summary 
            rpn_target_matchs, rpn_target_deltas = self.rpn.build_anchor_targets(gt_boxes, gt_class_ids, img_metas, imgs)
            # NOTE imgs just for summary
            rois, mrcnn_target_matchs, mrcnn_target_deltas, mrcnn_target_masks = self.detect_target.build_detect_targets(
                proposals, gt_boxes, gt_class_ids, gt_masks, img_metas, imgs)
            
        else:
            rois = proposals

        mrcnn_class_logits, mrcnn_probs, mrcnn_deltas = self.classifier_and_regression(
            [rois, mrcnn_feature_maps, img_metas], training=training)
        if not training:
            detections_list = self.classifier_and_regression.get_bboxes(mrcnn_probs, mrcnn_deltas, rois, img_metas)
            rois = tf.pad(tf.concat(detections_list, axis=0)[...,:4], [[0,0], [1,0]], constant_values=0)
        
        mrcnn_masks = self.masker([rois, mrcnn_feature_maps, img_metas], training=training)
        
        if training:
            rpn_class_loss, rpn_bbox_loss = self.rpn.loss(
                rpn_class_logits, rpn_deltas, rpn_target_matchs, rpn_target_deltas)
            mrcnn_class_loss, mrcnn_bbox_loss = self.classifier_and_regression.loss(
                mrcnn_class_logits, mrcnn_deltas, 
                mrcnn_target_matchs, mrcnn_target_deltas)
            mrcnn_mask_loss = self.masker.loss(mrcnn_target_masks, mrcnn_target_matchs, mrcnn_masks)
            return {"rpn_class_loss":   rpn_class_loss*self.config.LOSS_WEIGHTS["rpn_class_loss"], 
                    "rpn_bbox_loss":    rpn_bbox_loss*self.config.LOSS_WEIGHTS["rpn_bbox_loss"], 
                    "mrcnn_class_loss": mrcnn_class_loss*self.config.LOSS_WEIGHTS["mrcnn_class_loss"], 
                    "mrcnn_bbox_loss":  mrcnn_bbox_loss*self.config.LOSS_WEIGHTS["mrcnn_bbox_loss"],
                    "mrcnn_mask_loss":  mrcnn_mask_loss*self.config.LOSS_WEIGHTS["mrcnn_mask_loss"]}
        else:
            return detections_list, [mrcnn_masks]
    

    def unmold_detections(self, detections_list, masks, img_metas):
        return [
            self._unmold_single_detection(detections_list[i], masks[i], img_metas[i])
            for i in range(img_metas.shape[0])
        ]

    def _unmold_single_detection(self, detections, masks, img_meta):
        zero_ix = tf.where(tf.not_equal(detections[:, 4], 0))
        detections = tf.gather_nd(detections, zero_ix)

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:, :4]
        class_ids = tf.cast(detections[:, 4], tf.int32)
        scores = detections[:, 5]
        
        boxes = BoxTransforms().bbox_mapping_back(boxes, img_meta)
        ori_img_shape = parse_image_meta(img_meta)["ori_shape"]
        full_mask_list = []
        # masks = masks[tf.range(masks.shape), 
        # for i in range(boxes.shape[0]):
        #     full_mask = self.masker.unmold_mask(masks[i, :, :, class_ids[i]], boxes[i], ori_img_shape)
        #     full_mask_list.append(full_mask)
        # masks = tf.stack(full_mask_list, axis=2)
        return {'rois': boxes.numpy(),
                'class_ids': class_ids.numpy(),
                'scores': scores.numpy(),
                #'masks': masks.numpy()
                }

# =================================================
#
# =================================================
def trim_zeros(boxes, name=None):
    '''Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.
    
    Args
    ---
        boxes: [N, 4] matrix of boxes.
        non_zeros: [N] a 1D boolean mask identifying the rows to keep
    '''
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros

def parse_image_meta(meta):
    '''Parses a tensor that contains image attributes to its components.
    
    Args
    ---
        meta: [..., 11]

    Returns
    ---
        a dict of the parsed tensors.
    '''
    meta = meta.numpy()
    ori_shape = meta[..., 0:3]
    img_shape = meta[..., 3:6]
    pad_shape = meta[..., 6:9]
    scale = meta[..., 9]  
    flip = meta[..., 10]
    return {
        'ori_shape': ori_shape,
        'img_shape': img_shape,
        'pad_shape': pad_shape,
        'scale': scale,
        'flip': flip
    }

def calc_batch_padded_shape(meta):
    return tf.cast(tf.reduce_max(meta[:, 6:8], axis=0), tf.int32).numpy()

def calc_img_shapes(meta):
    return tf.cast(meta[..., 3:5], tf.int32).numpy()

def calc_pad_shapes(meta):
    return tf.cast(meta[..., 6:8], tf.int32).numpy()