# !/usr/bin/env python3.7
# -*- encoding: utf-8 -*-
# ***************************************************
# @File    :   model.py
# @Time    :   2019/05/24 22:45:25
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


import os
import time
import datetime
import re
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf
import h5py # load h5 model file
import imgaug
import scipy.misc as scm
import matplotlib.pyplot as plt
import inspect



from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM

from tensorflow.python.keras.engine import saving


from nets import utils

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.13")




############################################################
#  Feature Pyramid Network Heads
############################################################
class Classifier(keras.Model):
    def __init__(self, config, name="mrcnn_classifer_model", **kwargs):
        """
        Structures:
        -----------------------------------------------------------
            rois+featuremap
                  |
                RoiAlign
                  |
                CBRCBR
                  |
               Sequeeze
                /    \
            Dense    Dense
              |        |
            class     box
        """
        super(Classifier, self).__init__(name=name, **kwargs)
        self.config = config
        self.roialign = PyramidROIAlign([config.POOL_SIZE, config.POOL_SIZE], config, name="roi_align_classifier")
        self.tconv1 = KL.TimeDistributed(KL.Conv2D(config.FPN_CLASSIF_FC_LAYERS_SIZE, (config.POOL_SIZE, config.POOL_SIZE), padding="valid"), name="mrcnn_class_conv1")
        self.tnorm1 = KL.TimeDistributed(NORMALIZATION[config.NORM_TYPE](), name=f'mrcnn_class_{config.NORM_TYPE}1')
        self.relu1 = KL.Activation('relu')
        
        self.tconv2 = KL.TimeDistributed(KL.Conv2D(config.FPN_CLASSIF_FC_LAYERS_SIZE, (1,1)), name="mrcnn_class_conv2")
        self.tnorm2 = KL.TimeDistributed(NORMALIZATION[config.NORM_TYPE](), name=f'mrcnn_class_{config.NORM_TYPE}2')
        self.relu2 = KL.Activation('relu')

        self.reshape = KL.Reshape((-1, config.FPN_CLASSIF_FC_LAYERS_SIZE), name="pool_squeeze")
        self.tclass_logits = KL.TimeDistributed(KL.Dense(config.NUM_CLASSES), name='mrcnn_class_logits')
        self.tclass_probs = KL.TimeDistributed(KL.Activation("softmax"), name="mrcnn_class")
        self.tclass_logits2 = KL.TimeDistributed(KL.Dense(config.NUM_CLASSES2), name='mrcnn_class_logits2')
        self.tclass_probs2 = KL.TimeDistributed(KL.Activation("softmax"), name="mrcnn_class2")

        
        if config.MODEL == 'mrcnn':
            self.tbbox = KL.TimeDistributed(KL.Dense(config.NUM_CLASSES * 4, activation='linear'), name='mrcnn_bbox_fc')
            self.reshape_bbox = KL.Reshape((-1, config.NUM_CLASSES, 4), name="mrcnn_bbox")
        else:
            self.tbbox = KL.TimeDistributed(KL.Dense(4, activation='linear'), name='mrcnn_bbox_fc')
            self.reshape_bbox = KL.Reshape((-1, config.NUM_CLASSES, 4), name="mrcnn_bbox")
            self.concat = KL.Concatenate(axis=-2)

    def call(self, inputs, training=False):
        rois, image_meta, feature_maps = inputs
        x = self.roialign([rois, image_meta] + feature_maps)
        # Two 1024 FC layers (implemented with Conv2D for consistency)
        x = self.tconv1(x)
        x = self.tnorm1(x, training=training)
        x = self.relu1(x)
        x = self.tconv2(x)
        x = self.tnorm2(x, training=training)
        x = self.relu2(x)

        shared = self.reshape(x)
        # Classifier head
        mrcnn_class_logits = self.tclass_logits(shared)
        mrcnn_probs = self.tclass_probs(mrcnn_class_logits)
        if self.config.NUM_CLASSES2 > 2:
            # Classifier2 head
            mrcnn_class_logits2 = self.tclass_logits2(shared)
            mrcnn_probs2 = self.tclass_probs2(mrcnn_class_logits2)
        # BBox head
        # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
        x = self.tbbox(shared)
        # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
        mrcnn_bbox = self.reshape_bbox(x)
        if self.config.MODEL == 'siamese':
            mrcnn_bbox = self.concat([x for i in range(2)])
        mrcnn_box_outputs = {"mrcnn_class_logits": mrcnn_class_logits,
                            "mrcnn_probs": mrcnn_probs,
                            "mrcnn_bbox": mrcnn_bbox}
        if self.config.NUM_CLASSES2 > 2:
            mrcnn_box_outputs.update({
                "mrcnn_class_logits2": mrcnn_class_logits2,
                "mrcnn_probs2": mrcnn_probs2,
            })
        return mrcnn_box_outputs

class Mask(keras.Model):
    def __init__(self, config, name="mrcnn_mask_model", **kwargs):
        super(Mask, self).__init__(name=name, **kwargs)
        self.config = config
        self.roialign = PyramidROIAlign([config.MASK_POOL_SIZE, config.MASK_POOL_SIZE], config, name="roi_align_mask")
        for i in range(1, 5):
            setattr(self, f"mrcnn_mask_conv{i}", KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name=f"mrcnn_mask_conv{i}"))
            setattr(self, f"mrcnn_mask_{config.NORM_TYPE}{i}", KL.TimeDistributed(NORMALIZATION[config.NORM_TYPE](), name=f"mrcnn_mask_{config.NORM_TYPE}{i}"))
            setattr(self, f"relu{i}", KL.Activation('relu'))
        self.tdeconv = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"), name="mrcnn_mask_deconv")
        if config.MODEL == "mrcnn":
            self.tmask = KL.TimeDistributed(KL.Conv2D(config.NUM_CLASSES, (1, 1), strides=1, activation="sigmoid"), name="mrcnn_mask")
        else:
            self.tmask = KL.TimeDistributed(KL.Conv2D(1, (1, 1), strides=1, activation="sigmoid"), name="mrcnn_mask")
            self.concat = KL.Concatenate(axis=-1)
    def call(self, inputs, training=False):
        # ROI Pooling
        # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
        rois, image_meta, feature_maps = inputs
        x = self.roialign([rois, image_meta] + feature_maps)
        # Conv layers
        for i in range(1, 5):
            x = getattr(self, f"mrcnn_mask_conv{i}")(x)
            x = getattr(self, f"mrcnn_mask_{self.config.NORM_TYPE}{i}")(x, training=training)
            x = getattr(self, f"relu{i}")(x)
        x = self.tdeconv(x)
        x = self.tmask(x)
        if self.config.MODEL == "smrcnn":
            x = self.concat([x for i in range(2)])
        return x

class Recognition(keras.Model):
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

class L1_distance(keras.Model):
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


############################################################
#  Data Generator
############################################################
def load_image_gt(dataset, config, image_id, augmentation=None):
    """Load and return ground truth data for an image (image, mask, bounding boxes).
    Params:
    -----------------------------------------------------------
        augmentation:  Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
                       For example, passing imgaug.augmenters.Fliplr(0.5) flips images
                       right/left 50% of the time.
        config.USE_: If False, returns full-size masks that are the same height
                       and width as the original image. These can be big, for example
                       1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
                       224x224 and are generated by extracting the bounding box of the
                       object and resizing it to MINI_MASK_SHAPE.
    Returns:
    -----------------------------------------------------------
        image:     [height, width, 3]
        shape:     the original shape of the image before resizing and cropping.
        class_ids: [instance_count] Integer class IDs
        bbox:      [instance_count, (y1, x1, y2, x2)]
        mask:      [height, width, instance_count]. The height and width are those
                   of the image unless config.USE_ is True, in which case they are
                   defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    global_mask, mask, class_ids, class_ids2, text_embeds, embed_lengths = dataset.load_mask(image_id)
    original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    # TODO
    # global_mask = utils.resize_mask(global_mask, scale, padding, crop)
    mask = utils.resize_mask(mask, scale, padding, crop)

    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # global_mask = det.augment_image(global_mask)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        if config.SOFT_MASK:
            mask *= 255
        mask = det.augment_image(mask.astype(np.uint8),
                                hooks=imgaug.HooksImages(activator=hook))
        global_mask = det.augment_image(global_mask.astype(np.uint8),
                                hooks=imgaug.HooksImages(activator=hook))
        
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        if not config.SOFT_MASK:
            mask = mask.astype(np.bool)
            global_mask = global_mask.astype(np.bool)
        else:
            mask = np.array(mask/255., np.float32)


    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    
    class_ids = class_ids[_idx]
    class_ids2 = class_ids2[_idx]
    # NOTE NOTE NOTE if label2 is derection, augmentation mast be care hare
    # ------------------------------------------------------------
    def rot90_augment(image, mask, global_mask, class_ids2):
        k = np.random.choice([0, 1, 2, 3])
        if k:
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)
            global_mask = np.rot90(global_mask, k)
            map_dict = {1: dict(zip([0,1,2,3], [1,2,3,0])),
                        2: dict(zip([0,1,2,3], [2,3,0,1])),
                        3: dict(zip([0,1,2,3], [3,0,1,2]))}
            class_ids2 = np.array([map_dict[k][i] for i in class_ids2])
        return image, mask, global_mask, class_ids2
    image, mask, global_mask, class_ids2 = rot90_augment(image, mask, global_mask, class_ids2)
    text_embeds = text_embeds[_idx]
    embed_lengths = embed_lengths[_idx]
    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox, mask_score = utils.extract_bboxes(mask)
    rbbox = utils.extract_minienclose_bboxes(mask)
    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1
    # print ("dataset.source_class_ids", dataset.source_class_ids)
    # dataset.source_class_ids {'': [0], 'coco_label2': [0, 8, 9, 10, 11], 'coco': [0, 1, 2, 3, 4, 5, 6, 7]}
    source_class_ids2 = dataset.source_class_ids['coco_label2']
    active_class_ids[source_class_ids2[1: ]] = 1
    active_class_ids2 = active_class_ids[config.NUM_CLASSES: ]
    active_class_ids = active_class_ids[: config.NUM_CLASSES]
    
    # Resize masks to smaller size to reduce memory usage
    if config.USE_MINI_MASK:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE, softmask=config.SOFT_MASK)

    # Image meta data
    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale, active_class_ids, active_class_ids2)


    return image, image_meta, class_ids, class_ids2, bbox, rbbox, global_mask, mask, mask_score, text_embeds, embed_lengths


def build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, gt_masks_score, 
                            gt_class_ids2=None, gt_text_embed=None, gt_embed_length=None, config=None):
    """Generate targets for training Stage 2 classifier and mask heads.
    This is not used in normal training. It's useful for debugging or to train
    the Mask RCNN heads without using the RPN head.
    Params:
    -----------------------------------------------------------
        rpn_rois:        [N, (y1, x1, y2, x2)] proposal boxes.
        gt_class_ids:    [instance count] Integer class IDs
        gt_boxes:        [instance count, (y1, x1, y2, x2)]
        gt_masks:        [height, width, instance count] Ground truth masks. Can be full
                         size or mini-masks.
        gt_text_embed:   [instance count, embed_length] 
        gt_text_length:  [instance count]
        gt_embed_length: [instance count]
    Returns:
    -----------------------------------------------------------
        rois:         [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
        class_ids:    [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
        class_ids2:   [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
        bboxes:       [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (y, x, log(h), log(w))]. Class-specific
                      bbox refinements.
        masks:        [TRAIN_ROIS_PER_IMAGE, height, width, NUM_CLASSES). Class specific masks cropped
                      to bbox boundaries and resized to neural network output size.
        text_embed:   [TRAIN_ROIS_PER_IMAGE, embed_length] 
        text_length:  [TRAIN_ROIS_PER_IMAGE]
        embed_length: [TRAIN_ROIS_PER_IMAGE]
    """
    assert rpn_rois.shape[0] > 0
    assert gt_class_ids.dtype == np.int32, f"Expected int but got {gt_class_ids.dtype}"
    assert gt_boxes.dtype == np.int32, f"Expected int but got {gt_boxes.dtype}"
    if config.SOFT_MASK:
        assert gt_masks.dtype == np.float32, f"Expected float but got {gt_masks.dtype}"
    else:
        assert gt_masks.dtype == np.bool_, f"Expected bool but got {gt_masks.dtype}"
    assert gt_text_embed.dtype == np.int32, f"Expected int but got {gt_text_embed.dtype}"
    assert gt_embed_length.dtype == np.int32, f"Expected int but got {gt_embed_length.dtype}"
    # It's common to add GT Boxes to ROIs but we don't do that here because
    # according to XinLei Chen's paper, it doesn't help.

    # Trim empty padding in gt_boxes and gt_masks parts
    instance_ids = np.where(gt_class_ids > 0)[0]
    assert instance_ids.shape[0] > 0, "Image must contain instances."
    gt_class_ids = gt_class_ids[instance_ids]
    gt_boxes = gt_boxes[instance_ids]
    gt_masks = gt_masks[:, :, instance_ids]
    gt_masks_score = gt_masks_score[:, :, instance_ids]
    gt_class_ids2 = gt_class_ids2[instance_ids]
    gt_text_embed = gt_text_embed[instance_ids]
    gt_embed_length = gt_embed_length[instance_ids]
    # Compute areas of ROIs and ground truth boxes.
    rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * \
        (rpn_rois[:, 3] - rpn_rois[:, 1])
    gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
        (gt_boxes[:, 3] - gt_boxes[:, 1])

    # Compute overlaps [rpn_rois, gt_boxes]
    overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
    for i in range(overlaps.shape[1]):
        gt = gt_boxes[i]
        overlaps[:, i] = utils.compute_iou(
            gt, rpn_rois, gt_box_area[i], rpn_roi_area)

    # Assign ROIs to GT boxes
    rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)
    rpn_roi_iou_max = overlaps[np.arange(overlaps.shape[0]), rpn_roi_iou_argmax]
    # GT box assigned to each ROI
    rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]
    rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]
    rpn_roi_gt_class_ids2 = gt_class_ids2[rpn_roi_iou_argmax]
    rpn_roi_gt_text_embed = gt_text_embed[rpn_roi_iou_argmax]
    rpn_roi_gt_embed_length = gt_embed_length[rpn_roi_iou_argmax]

    # Positive ROIs are those with >= 0.5 IoU with a GT box.
    fg_ids = np.where(rpn_roi_iou_max > 0.5)[0]

    # Negative ROIs are those with max IoU 0.1-0.5 (hard example mining)
    # TODO: To hard example mine or not to hard example mine, that's the question
    # bg_ids = np.where((rpn_roi_iou_max >= 0.1) & (rpn_roi_iou_max < 0.5))[0]
    bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]

    # Subsample ROIs. Aim for 33% foreground.
    # FG
    fg_roi_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    if fg_ids.shape[0] > fg_roi_count:
        keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
    else:
        keep_fg_ids = fg_ids
    # BG
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep_fg_ids.shape[0]
    if bg_ids.shape[0] > remaining:
        keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
    else:
        keep_bg_ids = bg_ids
    # Combine indices of ROIs to keep
    keep = np.concatenate([keep_fg_ids, keep_bg_ids])
    # Need more?
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep.shape[0]
    if remaining > 0:
        # Looks like we don't have enough samples to maintain the desired
        # balance. Reduce requirements and fill in the rest. This is
        # likely different from the Mask RCNN paper.

        # There is a small chance we have neither fg nor bg samples.
        if keep.shape[0] == 0:
            # Pick bg regions with easier IoU threshold
            bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
            assert bg_ids.shape[0] >= remaining
            keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
            assert keep_bg_ids.shape[0] == remaining
            keep = np.concatenate([keep, keep_bg_ids])
        else:
            # Fill the rest with repeated bg rois.
            keep_extra_ids = np.random.choice(
                keep_bg_ids, remaining, replace=True)
            keep = np.concatenate([keep, keep_extra_ids])
    assert keep.shape[0] == config.TRAIN_ROIS_PER_IMAGE, \
            f"keep doesn't match ROI batch size {keep.shape[0]}, {config.TRAIN_ROIS_PER_IMAGE}"

    # Reset the gt boxes assigned to BG ROIs.
    rpn_roi_gt_boxes[keep_bg_ids, :] = 0
    rpn_roi_gt_class_ids[keep_bg_ids] = 0
    rpn_roi_gt_class_ids2[keep_bg_ids] = 0
    rpn_roi_gt_text_embed[keep_bg_ids] = [0]*config.MAX_LABEL_LENGTH # TODO may be error
    rpn_roi_gt_embed_length[keep_bg_ids] = 0

    # For each kept ROI, assign a class_id, and for FG ROIs also add bbox refinement.
    rois = rpn_rois[keep]
    roi_gt_boxes = rpn_roi_gt_boxes[keep]
    roi_gt_class_ids = rpn_roi_gt_class_ids[keep]
    roi_gt_class_ids2 = rpn_roi_gt_class_ids2[keep]
    roi_gt_text_embed = rpn_roi_gt_text_embed[keep]
    roi_gt_embed_length = rpn_roi_gt_embed_length[keep]
    roi_gt_assignment = rpn_roi_iou_argmax[keep]

    # Class-aware bbox deltas. [y, x, log(h), log(w)]
    bboxes = np.zeros((config.TRAIN_ROIS_PER_IMAGE,
                       config.NUM_CLASSES, 4), dtype=np.float32)
    pos_ids = np.where(roi_gt_class_ids > 0)[0]
    bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = utils.box_refinement(
        rois[pos_ids], roi_gt_boxes[pos_ids, :4])
    # Normalize bbox refinements
    bboxes /= config.BBOX_STD_DEV

    # Generate class-specific target masks
    masks = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0], config.MASK_SHAPE[1], config.NUM_CLASSES),
                     dtype=np.float32)
    for i in pos_ids:
        class_id = roi_gt_class_ids[i]
        assert class_id > 0, "class id must be greater than 0"
        gt_id = roi_gt_assignment[i]
        class_mask = gt_masks[:, :, gt_id]

        if config.USE_MINI_MASK:
            # Create a mask placeholder, the size of the image
            if config.SOFT_MASK:
                placeholder = np.zeros(config.IMAGE_SHAPE[:2], dtype=np.float32)
            else:
                placeholder = np.zeros(config.IMAGE_SHAPE[:2], dtype=bool)
            # GT box
            gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[gt_id]
            gt_w = gt_x2 - gt_x1
            gt_h = gt_y2 - gt_y1
            # Resize mini mask to size of GT box
            if config.SOFT_MASK:
                placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = utils.resize(class_mask, (gt_h, gt_w))
            else:
                placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = np.round(utils.resize(class_mask, (gt_h, gt_w))).astype(bool)
            # Place the mini batch in the placeholder
            class_mask = placeholder

        # Pick part of the mask and resize it
        y1, x1, y2, x2 = rois[i].astype(np.int32)
        m = class_mask[y1:y2, x1:x2]
        mask = utils.resize(m, config.MASK_SHAPE)
        masks[i, :, :, class_id] = mask


    return rois, roi_gt_class_ids, roi_gt_class_ids2, bboxes, masks, roi_gt_text_embed, roi_gt_embed_length

def get_one_target(category, dataset, config, augmentation=None, target_size_limit=0, max_attempts=10, return_all=False, return_original_size=False):

    n_attempts = 0
    while True:
        # Get index with corresponding images for each category
        category_image_index = dataset.category_image_index
        # Draw a random image
        random_image_id = np.random.choice(category_image_index[category])
        # Load image   
        target_image, target_image_meta, target_class_ids, target_class_ids2, target_bboxes,\
            target_rboxes, target_global_mask, target_mask, target_mask_score, target_text_embeds,\
            target_embed_lengths = load_image_gt(dataset, config, random_image_id, augmentation=augmentation)
        # target_image, target_image_meta, target_class_ids, target_boxes, target_masks = \
        #     load_image_gt(dataset, config, random_image_id, augmentation=augmentation)
        # print(random_image_id, category, target_class_ids)
        if not np.any(target_class_ids == category):
            continue

        # try:
        #     box_ind = np.random.choice(np.where(target_class_ids == category)[0])   
        # except ValueError:
        #     return None
        box_ind = np.random.choice(np.where(target_class_ids == category)[0])   
        tb = target_bboxes[box_ind,:]
        target = target_image[tb[0]:tb[2],tb[1]:tb[3],:]
        original_size = target.shape
        target, window, scale, padding, crop = utils.resize_image(
            target,
            min_dim=config.TARGET_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE, #Same scaling as the image
            max_dim=config.TARGET_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE) #Same output format as the image

        n_attempts = n_attempts + 1
        if (min(original_size[:2]) >= target_size_limit) or (n_attempts >= max_attempts):
            break
    
    if return_all:
        return target, window, scale, padding, crop
    elif return_original_size:
        return target, original_size
    else:
        return target

def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.
    Params:
    -----------------------------------------------------------
        anchors:      [num_anchors, (y1, x1, y2, x2)]
        gt_class_ids: [num_gt_boxes] Integer class IDs.
        gt_boxes:     [num_gt_boxes, (y1, x1, y2, x2)]
    Returns:
    -----------------------------------------------------------
        rpn_match: [N] (int32) matches between anchors and GT boxes.
                   1 = positive anchor, -1 = negative anchor, 0 = neutral
        rpn_bbox:  [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE - np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox


def generate_random_rois(image_shape, count, gt_class_ids, gt_boxes):
    """Generates ROI proposals similar to what a region proposal network
    would generate.
    Params:
    -----------------------------------------------------------
        image_shape:  [Height, Width, Depth]
        count:        Number of ROIs to generate
        gt_class_ids: [N] Integer ground truth class IDs
        gt_boxes:     [N, (y1, x1, y2, x2)] Ground truth boxes in pixels.
        Returns:      [count, (y1, x1, y2, x2)] ROI boxes in pixels.
    """
    # placeholder
    rois = np.zeros((count, 4), dtype=np.int32)

    # Generate random ROIs around GT boxes (90% of count)
    rois_per_box = int(0.9 * count / gt_boxes.shape[0])
    for i in range(gt_boxes.shape[0]):
        gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
        h = gt_y2 - gt_y1
        w = gt_x2 - gt_x1
        # random boundaries
        r_y1 = max(gt_y1 - h, 0)
        r_y2 = min(gt_y2 + h, image_shape[0])
        r_x1 = max(gt_x1 - w, 0)
        r_x2 = min(gt_x2 + w, image_shape[1])

        # To avoid generating boxes with zero area, we generate double what
        # we need and filter out the extra. If we get fewer valid boxes
        # than we need, we loop and try again.
        while True:
            y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
            x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box * 2, 2))
            # Filter out zero area boxes
            threshold = 1
            y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >= threshold][:rois_per_box]
            x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >= threshold][:rois_per_box]
            if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
                break

        # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
        # into x1, y1, x2, y2 order
        x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
        y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
        box_rois = np.hstack([y1, x1, y2, x2])
        rois[rois_per_box * i:rois_per_box * (i + 1)] = box_rois

    # Generate random ROIs anywhere in the image (10% of count)
    remaining_count = count - (rois_per_box * gt_boxes.shape[0])
    # To avoid generating boxes with zero area, we generate double what
    # we need and filter out the extra. If we get fewer valid boxes
    # than we need, we loop and try again.
    while True:
        y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
        x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
        # Filter out zero area boxes
        threshold = 1
        y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >= threshold][:remaining_count]
        x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >= threshold][:remaining_count]
        if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
            break

    # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
    # into x1, y1, x2, y2 order
    x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
    y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
    global_rois = np.hstack([y1, x1, y2, x2])
    rois[-remaining_count:] = global_rois
    return rois

class DataLoader():
    def __init__(self, dataset, config, augmentation=None, detection_targets=False):
        self.dataset = dataset
        self.config = config
        self.augmentation = augmentation
        self.detection_targets = detection_targets
        # self.on_epoch_end()

    @staticmethod
    def load_image_gt(dataset, config, image_id, augmentation=None):
        """Load and return ground truth data for an image (image, mask, bounding boxes).
        Params:
        -----------------------------------------------------------
            augmentation:  Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
                        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
                        right/left 50% of the time.
            config.USE_: If False, returns full-size masks that are the same height
                        and width as the original image. These can be big, for example
                        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
                        224x224 and are generated by extracting the bounding box of the
                        object and resizing it to MINI_MASK_SHAPE.
        Returns:
        -----------------------------------------------------------
            image:     [height, width, 3]
            shape:     the original shape of the image before resizing and cropping.
            class_ids: [instance_count] Integer class IDs
            bbox:      [instance_count, (y1, x1, y2, x2)]
            mask:      [height, width, instance_count]. The height and width are those
                    of the image unless config.USE_ is True, in which case they are
                    defined in MINI_MASK_SHAPE.
        """
        # Load image and mask
        image = dataset.load_image(image_id)
        global_mask, mask, class_ids, class_ids2, text_embeds, embed_lengths = dataset.load_mask(image_id)
        original_shape = image.shape
        image, window, scale, padding, crop = utils.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE,
            max_dim=config.IMAGE_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE)
        # TODO
        # global_mask = utils.resize_mask(global_mask, scale, padding, crop)
        mask = utils.resize_mask(mask, scale, padding, crop)

        # Augmentation
        # This requires the imgaug lib (https://github.com/aleju/imgaug)
        if augmentation and dataset.image_info[image_id]['source'] not in config.NO_AUGMENT_SOURCES:
            # Augmenters that are safe to apply to masks
            # Some, such as Affine, have settings that make them unsafe, so always
            # test your augmentation on masks
            MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                            "Fliplr", "Flipud", "CropAndPad",
                            "Affine", "PiecewiseAffine"]

            def hook(images, augmenter, parents, default):
                """Determines which augmenters to apply to masks."""
                return augmenter.__class__.__name__ in MASK_AUGMENTERS

            # Store shapes before augmentation to compare
            image_shape = image.shape
            mask_shape = mask.shape
            # Make augmenters deterministic to apply similarly to images and masks
            det = augmentation.to_deterministic()
            image = det.augment_image(image)
            # global_mask = det.augment_image(global_mask)
            # Change mask to np.uint8 because imgaug doesn't support np.bool
            if config.SOFT_MASK:
                mask *= 255
            mask = det.augment_image(mask.astype(np.uint8),
                                    hooks=imgaug.HooksImages(activator=hook))
            global_mask = det.augment_image(global_mask.astype(np.uint8),
                                    hooks=imgaug.HooksImages(activator=hook))
            
            # Verify that shapes didn't change
            assert image.shape == image_shape, "Augmentation shouldn't change image size"
            assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
            # Change mask back to bool
            if not config.SOFT_MASK:
                mask = mask.astype(np.bool)
                global_mask = global_mask.astype(np.bool)
            else:
                mask = np.array(mask/255., np.float32)


        # Note that some boxes might be all zeros if the corresponding mask got cropped out.
        # and here is to filter them out
        _idx = np.sum(mask, axis=(0, 1)) > 0
        mask = mask[:, :, _idx]
        
        class_ids = class_ids[_idx]
        class_ids2 = class_ids2[_idx]
        # NOTE NOTE NOTE if label2 is derection, augmentation mast be care hare
        # ------------------------------------------------------------
        def rot90_augment(image, mask, global_mask, class_ids2):
            k = random.choice([0, 1, 2, 3])
            if k:
                image = np.rot90(image, k)
                mask = np.rot90(mask, k)
                global_mask = np.rot90(global_mask, k)
                map_dict = {1: dict(zip([0,1,2,3], [1,2,3,0])),
                            2: dict(zip([0,1,2,3], [2,3,0,1])),
                            3: dict(zip([0,1,2,3], [3,0,1,2]))}
                class_ids2 = np.array([map_dict[k][i] for i in class_ids2])
            return image, mask, global_mask, class_ids2
        image, mask, global_mask, class_ids2 = rot90_augment(image, mask, global_mask, class_ids2)
        text_embeds = text_embeds[_idx]
        embed_lengths = embed_lengths[_idx]
        # Bounding boxes. Note that some boxes might be all zeros
        # if the corresponding mask got cropped out.
        # bbox: [num_instances, (y1, x1, y2, x2)]
        bbox, mask_score = utils.extract_bboxes(mask)
        rbbox = utils.extract_minienclose_bboxes(mask)
        # Active classes
        # Different datasets have different classes, so track the
        # classes supported in the dataset of this image.
        active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
        source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
        active_class_ids[source_class_ids] = 1
        # print ("dataset.source_class_ids", dataset.source_class_ids)
        # dataset.source_class_ids {'': [0], 'coco_label2': [0, 8, 9, 10, 11], 'coco': [0, 1, 2, 3, 4, 5, 6, 7]}
        source_class_ids2 = dataset.source_class_ids['coco_label2']
        active_class_ids[source_class_ids2[1: ]] = 1
        active_class_ids2 = active_class_ids[config.NUM_CLASSES: ]
        active_class_ids = active_class_ids[: config.NUM_CLASSES]
        
        # Resize masks to smaller size to reduce memory usage
        if config.USE_MINI_MASK:
            mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE, softmask=config.SOFT_MASK)

        # Image meta data
        image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                        window, scale, active_class_ids, active_class_ids2)


        return image, image_meta, class_ids, class_ids2, bbox, rbbox, global_mask, mask, mask_score, text_embeds, embed_lengths


    def __len__(self):
        return len(self.dataset.image_ids)
    
    def __getitem__(self, idx):
        '''Load the image and its bboxes for the given index.
        
        Args
        ---
            idx: the index of images.
            
        Returns
        ---
            tuple: A tuple containing the following items: image, 
                bboxes, labels.
        '''
        # Increment index to pick next image. Shuffle if at the start of an epoch.
        # image_index = (image_index + 1) % len(image_ids)
        # if shuffle and image_index == 0:
        #     np.random.shuffle(image_ids)

        # # Get GT bounding boxes and masks for image.
        # image_id = image_ids[image_index]
        image, image_meta, class_ids, class_ids2, bbox, rbox, global_mask, mask, mask_score, text_embeds, embed_lengths = \
            self.load_image_gt(self.dataset, self.config, idx, self.augmentation)
        # Skip images that have no instances. This can happen in cases
        # where we train on a subset of classes and the image doesn't
        # have any of the classes we care about.
        assert np.any(class_ids > 0), "No instances"
        image = mold_image(image.astype(np.float32), self.config)
        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]
        anchors = build_anchors(self.config, to_tensor=False)
        # RPN Targets
        # if rpn have muiltple label, rewrite here
        rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors, class_ids, bbox, self.config)
        rpn_match = rpn_match[:, np.newaxis]
        # Mask R-CNN Targets
        if not self.config.USE_RPN_ROIS:
            rpn_rois = generate_random_rois(image.shape, 10000, class_ids, bbox)
            if self.detection_targets:
                rois, mrcnn_class_ids, mrcnn_class_ids2, mrcnn_bbox, mrcnn_rbbox, mrcnn_mask,\
                    mrcnn_text_embeds, mrcnn_embed_lengths = build_detection_targets(
                        rpn_rois, class_ids, bbox, rbox, mask, mask_score, self.config)
        return image, image_meta, class_ids, class_ids2, bbox, rbox, global_mask, \
            mask, mask_score, text_embeds, embed_lengths, rpn_match, rpn_bbox

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dataset.image_ids))
        # if self.shuffle == True:
        np.random.shuffle(self.indexes)

class DataGenerator(object):
    def __init__(self, dataset, shuffle=False):
        self.dataset = dataset
        self.shuffle = shuffle
    
    def __call__(self):
        indices = np.arange(len(self.dataset))
        # print ("len", len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
        for img_idx in indices:
            # print (img_idx)
            data = self.dataset[img_idx]
            yield tuple(data)



def data_generator(dataset, config, shuffle=True, augmentation=None,
                   random_rois=0, batch_size=1, detection_targets=False,
                   diverse=0, no_augmentation_sources=None):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.
    Params:
    -----------------------------------------------------------
        dataset:                 The Dataset object to pick data from
        config:                  The model config object
        shuffle:                 If True, shuffles the samples before every epoch
        augmentation:            Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
                                 For example, passing imgaug.augmenters.Fliplr(0.5) flips images
                                 right/left 50% of the time.
        random_rois:             If > 0 then generate proposals to be used to train the
                                 network classifier and mask heads. Useful if training
                                 the Mask RCNN part without the RPN.
        batch_size:              How many images to return in each call
        detection_targets:       If True, generate detection targets (class IDs, bbox
                                 deltas, and masks). Typically for debugging or visualizations because
                                 in trainig detection targets are generated by DetectionTargetLayer.
        diverse:                 Float in [0,1] indicatiing probability to draw a target
                                 from any random class instead of one from the image classes
        no_augmentation_sources: Optional. List of sources to exclude for
                                 augmentation. A source is string that identifies a dataset and is
                                 defined in the Dataset class.
    Returns:
    -----------------------------------------------------------
    a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The contents
    of the lists differs depending on the received arguments:
        inputs list:
            - images:       [batch, H, W, C]
            - image_meta:   [batch, (meta data)] Image details. See compose_image_meta()
            - rpn_match:    [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
            - rpn_bbox:     [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
            - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
            - gt_boxes:     [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
            - gt_masks:     [batch, height, width, MAX_GT_INSTANCES]. The height and width
                            are those of the image unless use_mini_mask is True, in which
                            case they are defined in MINI_MASK_SHAPE.
        outputs list:   Usually empty in regular training. But if detection_targets
                        is True then the outputs list contains target class_ids, bbox deltas,
                        and masks.
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0
    no_augmentation_sources = no_augmentation_sources or []

    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    # Keras requires a generator to run indefinitely.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]
            # If the image source is not to be augmented pass None as augmentation
            if dataset.image_info[image_id]['source'] in no_augmentation_sources: augmentation = None
            image, image_meta, gt_class_ids, gt_class_ids2, gt_boxes, gt_rboxes, gt_global_mask, \
                gt_masks, gt_mask_score, gt_text_embeds, gt_embed_lengths = load_image_gt(dataset, config, image_id,
                            augmentation=augmentation)

            
            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue

            # Use only positive class_ids
            categories = np.unique(gt_class_ids)
            _idx = categories > 0
            categories = categories[_idx]
            
            if config.MODEL == "smrcnn":
                # Use only active classes
                active_categories = []
                for c in categories:
                    if any(c == dataset.ACTIVE_CLASSES):
                        active_categories.append(c)
                
                # Skiop image if it contains no instance of any active class    
                if not np.any(np.array(active_categories) > 0):
                    continue
                # Randomly select category
                category = np.random.choice(active_categories)
                    
                # NOTE for siamese
                # Generate siamese target crop
                targets = []
                for i in range(config.NUM_TARGETS):
                    targets.append(get_one_target(category, dataset, config, augmentation=augmentation))
                # target = np.stack(target, axis=0)
                        
                # print(target_class_id)
                target_class_id = category
                target_class_ids = np.array([target_class_id])
                
                idx = gt_class_ids == target_class_id
                siamese_class_ids = idx.astype('int8')
                # print(idx)
                # print(gt_boxes.shape, gt_masks.shape)
                siamese_class_ids = siamese_class_ids[idx]
                gt_class_ids = gt_class_ids[idx]
                gt_boxes = gt_boxes[idx,:]
                gt_masks = gt_masks[:,:,idx]
                image_meta = image_meta[:15] # TODO
                # --------------------------------------------------------------

            # RPN Targets
            # if rpn have muiltple label, rewrite here
            rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
                                                    gt_class_ids, gt_boxes, config)

            # Mask R-CNN Targets
            if random_rois:
                rpn_rois = generate_random_rois(image.shape, random_rois, gt_class_ids, gt_boxes)
                if detection_targets:
                    rois, mrcnn_class_ids, mrcnn_class_ids2, mrcnn_bbox, mrcnn_rbbox, mrcnn_mask,\
                        mrcnn_text_embeds, mrcnn_embed_lengths = build_detection_targets(
                            rpn_rois, gt_class_ids, gt_boxes, gt_rboxes, gt_masks, gt_mask_score, gt_class_ids2, config)

            # Init batch arrays
            if b == 0:
                batch_image_meta = np.zeros((batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros([batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros([batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros((batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros((batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros((batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                batch_gt_rboxes = np.zeros((batch_size, config.MAX_GT_INSTANCES, 5), dtype=np.float32)
                if config.MODEL == "smrcnn":
                    batch_targets = np.zeros((batch_size, config.NUM_TARGETS) + targets[0].shape, dtype=np.float32)
                batch_gt_masks = np.zeros((batch_size, gt_masks.shape[0], gt_masks.shape[1],
                     config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)
                batch_gt_class_ids2 = np.zeros((batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_text_embeds = np.zeros((batch_size, config.MAX_GT_INSTANCES, config.MAX_LABEL_LENGTH), dtype=np.int32)
                batch_gt_embed_lengths = np.zeros((batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                if random_rois:
                    batch_rpn_rois = np.zeros((batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                    if detection_targets:
                        batch_rois = np.zeros((batch_size,) + rois.shape, dtype=rois.dtype)
                        batch_mrcnn_class_ids = np.zeros((batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
                        
                        # ************************* NOTE for 2 label dataset
                        if config.NUM_CLASSES2 > 2:
                            batch_mrcnn_class_ids2 = np.zeros(
                                (batch_size,) + mrcnn_class_ids2.shape, dtype=mrcnn_class_ids.dtype)
                        # ************************* NOTE for ocr
                        if config.READ:
                            batch_mrcnn_text_embeds = np.zeros(
                                (batch_size,) + mrcnn_text_embeds.shape, dtype=mrcnn_text_embeds.dtype)
                            batch_mrcnn_embed_lengths = np.zeros(
                                (batch_size,) + mrcnn_embed_lengths.shape, dtype=mrcnn_text_embeds.dtype)
                        batch_mrcnn_bbox = np.zeros((batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                        batch_mrcnn_rbbox = np.zeros((batch_size,) + mrcnn_rbbox.shape, dtype=mrcnn_rbbox.dtype)
                        batch_mrcnn_mask = np.zeros((batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                siamese_class_ids = siamese_class_ids[ids] # NOTE
                gt_boxes = gt_boxes[ids]
                gt_rboxes = gt_rboxes[ids]
                gt_masks = gt_masks[:, :, ids]
                gt_class_ids2 = gt_class_ids2[ids]
                gt_text_embeds = gt_text_embeds[ids]
                gt_embed_lengths = gt_embed_lengths[ids]

            # Add to batch
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = mold_image(image.astype(np.float32), config)
            # NOTE for siamese
            if config.MODEL == "smrcnn":
                batch_targets[b] = np.stack([mold_image(target.astype(np.float32), config) for target in targets], axis=0)
                batch_gt_class_ids[b, :siamese_class_ids.shape[0]] = siamese_class_ids
            else:
                batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_rboxes[b, :gt_rboxes.shape[0]] = gt_rboxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            batch_gt_class_ids2[b, :gt_class_ids2.shape[0]] = gt_class_ids2
            batch_gt_text_embeds[b, :gt_text_embeds.shape[0], :gt_text_embeds.shape[1]] = gt_text_embeds
            batch_gt_embed_lengths[b, :gt_embed_lengths.shape[0]] = gt_embed_lengths
            if random_rois:
                batch_rpn_rois[b] = rpn_rois
                if detection_targets:
                    batch_rois[b] = rois
                    batch_mrcnn_class_ids[b] = mrcnn_class_ids
                    batch_mrcnn_bbox[b] = mrcnn_bbox
                    batch_mrcnn_rbbox[b] = mrcnn_rbbox
                    batch_mrcnn_mask[b] = mrcnn_mask
                    batch_mrcnn_class_ids2[b] = mrcnn_class_ids2
                    batch_mrcnn_text_embeds[b] = mrcnn_text_embeds
                    batch_mrcnn_embed_lengths[b] = mrcnn_embed_lengths
            b += 1
            # Batch full?
            if b >= batch_size:
                

                # NOTE for siamese
                if config.MODEL == "smrcnn":
                    inputs = [batch_images, batch_image_meta, batch_targets, batch_rpn_match, batch_rpn_bbox,
                            batch_gt_class_ids, batch_gt_class_ids2, batch_gt_boxes, batch_gt_rboxes, batch_gt_masks,
                            batch_gt_text_embeds, batch_gt_embed_lengths]
                else:
                    inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                            batch_gt_class_ids, batch_gt_class_ids2, batch_gt_boxes, batch_gt_rboxes, batch_gt_masks,
                            batch_gt_text_embeds, batch_gt_embed_lengths]
                outputs = []
                if random_rois:
                    inputs.extend([batch_rpn_rois])
                    if detection_targets:
                        inputs.extend([batch_rois])
                        # Keras requires that output and targets have the same number of dimensions
                        batch_mrcnn_class_ids = np.expand_dims(batch_mrcnn_class_ids, -1)                       
                        
                        # ************************* NOTE for 2 label dataset
                        # ************************* NOTE for ocr
                        if config.RBOX and config.READ and config.HAVE_LABEL2:
                            batch_mrcnn_class_ids2 = np.expand_dims(batch_mrcnn_class_ids2, -1)
                            batch_mrcnn_text_embeds = np.expand_dims(batch_mrcnn_text_embeds, -1)
                            batch_mrcnn_embed_lengths = np.expand_dims(batch_mrcnn_embed_lengths, -1)
                            outputs.extend(
                                [batch_mrcnn_class_ids, batch_mrcnn_class_ids2, batch_mrcnn_bbox,\
                                    batch_mrcnn_rbbox, batch_mrcnn_mask,
                                batch_mrcnn_text_embeds, batch_mrcnn_embed_lengths])
                        elif config.RBOX and config.READ and not config.HAVE_LABEL2:
                            batch_mrcnn_text_embeds = np.expand_dims(batch_mrcnn_text_embeds, -1)
                            batch_mrcnn_embed_lengths = np.expand_dims(batch_mrcnn_embed_lengths, -1)
                            outputs.extend(
                                [batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_rbbox, batch_mrcnn_mask,
                                batch_mrcnn_text_embeds, batch_mrcnn_embed_lengths])
                        elif config.RBOX and not config.READ and config.HAVE_LABEL2:
                            batch_mrcnn_class_ids2 = np.expand_dims(batch_mrcnn_class_ids2, -1)  
                            outputs.extend(
                                [batch_mrcnn_class_ids, batch_mrcnn_class_ids2, batch_mrcnn_bbox,\
                                     batch_mrcnn_rbbox, batch_mrcnn_mask])
                        elif config.RBOX and not config.READ and not config.HAVE_LABEL2:
                            outputs.extend(
                                [batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_rbbox, batch_mrcnn_mask])
                        elif not config.RBOX and config.READ and config.HAVE_LABEL2:
                            batch_mrcnn_class_ids2 = np.expand_dims(batch_mrcnn_class_ids2, -1)
                            batch_mrcnn_text_embeds = np.expand_dims(batch_mrcnn_text_embeds, -1)
                            batch_mrcnn_embed_lengths = np.expand_dims(batch_mrcnn_embed_lengths, -1)
                            outputs.extend(
                                [batch_mrcnn_class_ids, batch_mrcnn_class_ids2, batch_mrcnn_bbox,\
                                    batch_mrcnn_mask,
                                batch_mrcnn_text_embeds, batch_mrcnn_embed_lengths])
                        elif not config.RBOX and config.READ and not config.HAVE_LABEL2:
                            batch_mrcnn_text_embeds = np.expand_dims(batch_mrcnn_text_embeds, -1)
                            batch_mrcnn_embed_lengths = np.expand_dims(batch_mrcnn_embed_lengths, -1)
                            outputs.extend(
                                [batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask,
                                batch_mrcnn_text_embeds, batch_mrcnn_embed_lengths])
                        elif not config.RBOX and not config.READ and config.HAVE_LABEL2:
                            batch_mrcnn_class_ids2 = np.expand_dims(batch_mrcnn_class_ids2, -1)  
                            outputs.extend(
                                [batch_mrcnn_class_ids, batch_mrcnn_class_ids2, batch_mrcnn_bbox,\
                                     batch_mrcnn_mask])
                        elif not config.RBOX and not config.READ and not config.HAVE_LABEL2:
                            outputs.extend(
                                [batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception(f"Error processing image {dataset.image_info[image_id]}")
            error_count += 1
            if error_count > 5:
                raise


class OOD(keras.Model):
    def __init__(self, mode, config, model_dir, **kwargs):
        super(self.__class__, self).__init__(name=config.NAME, **kwargs)
        """
        Params:
        -----------------------------------------------------------
            mode:      Either "training" or "inference"
            config:    A Sub-class of the Config class
            model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
        Params:
        -----------------------------------------------------------
            input_shape: The shape of the input image.
            mode:        Either "training" or "inference". The inputs and
                         outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        #  NOTE Inputs --------------------
        # input_image = KL.Input(shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
        input_image = KL.Input(shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], config.IMAGE_SHAPE[2]], name="input_image")
        input_target = KL.Input(shape=[config.NUM_TARGETS] + config.TARGET_SHAPE.tolist(), name="input_target")
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE], name="input_image_meta")
        if mode == "training":
            # NOTE RPN GT -----------------
            input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # NOTE RCNN GT ----------------
            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            input_gt_class_ids2 = KL.Input(shape=[None], name="input_gt_class_ids2", dtype=tf.int32)
            
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            input_gt_rboxes = KL.Input(shape=[None, 5], name="input_gt_rboxes", dtype=tf.float32)
            # Normalize coordinates
            gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(x, tf.shape(input_image)[1:3]))(input_gt_boxes)
            gt_rboxes = KL.Lambda(lambda x: norm_rboxes_graph(x, tf.shape(input_image)[1:3]))(input_gt_rboxes)
            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            mask_type = tf.float32 if config.SOFT_MASK else bool
            mask_shape = config.MINI_MASK_SHAPE if config.USE_MINI_MASK else config.IMAGE_SHAPE
            input_gt_masks = KL.Input(shape=[mask_shape[0], mask_shape[1], None], name="input_gt_masks", dtype=mask_type)
            input_gt_masks_score = KL.Input(shape=[None], name="input_gt_masks_score", dtype=tf.float32)

            # NOTE CRNN GT -------------------
            input_gt_recognition = KL.Input(shape=(config.READ_IMG_HEIGHT, config.READ_IMG_WIDTH, None),name='the_input',dtype=tf.float32)
            # fixed length, paded with 10000
            input_gt_text_embed = KL.Input(shape=[None, config.MAX_LABEL_LENGTH],name='input_gt_text_embed',dtype=tf.float32)
            # fixed length
            input_gt_text_length = KL.Input(shape=[None], name='input_gt_text_length', dtype=tf.int64)
            input_gt_embed_length = KL.Input(shape=[None], name='input_gt_embed_length', dtype=tf.int64)
            
        
        elif mode == "inference":
            # Anchors in normalized coordinates
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        # Build the shared convolutional layers.
        # ----------NOTE Bottom-up Layers NOTE -----------
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        # if callable(config.BACKBONE):
        #     _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True,
        #                                         train_bn=config.TRAIN_BN, norm_type=config.NORM_TYPE)
        # elif config.BACKBONE in ["mobilenet224v1"]:
        #     _, C2, C3, C4, C5 = mobilenet_graph(input_image, config.BACKBONE, 
        #                                         alpha=1.0, train_bn=config.TRAIN_BN, norm_type=config.NORM_TYPE)
        # elif config.BACKBONE in ["mnasnet"]:
        #     _, C2, C3, C4, C5 = mnasnet_graph(input_image, config.BACKBONE, 
        #                                         alpha=1.0, train_bn=config.TRAIN_BN, norm_type=config.NORM_TYPE)
        # elif config.BACKBONE in ["xception"]:
        #     _, C2, C3, C4, C5 = xception_graph(input_image, config.BACKBONE, 
        #                                        train_bn=config.TRAIN_BN, norm_type=config.NORM_TYPE)
        # elif config.BACKBONE in ["nasnet"]:
        #     _, C2, C3, C4, C5 = nasnet_graph(input_image, config.BACKBONE, 
        #                                     alpha=1, train_bn=config.TRAIN_BN, norm_type=config.NORM_TYPE)
        # else:
        #     _, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE,
        #                                      stage5=True, train_bn=config.TRAIN_BN, norm_type=config.NORM_TYPE)
        C2, C3, C4, C5 = ResNet(101, name='resnet101')(input_image)
        # ----------NOTE Top-down Layers NOTE -----------
        # TODO: add assert to varify feature map sizes match what's in config
        # Common FPN ----------------------------------------------------------------
        P2, P3, P4, P5, P6 = FPN(config.TOP_DOWN_PYRAMID_SIZE)([C2, C3, C4, C5])

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]
        
        if config.MODEL == "smrcnn":
            # Create Target FR
            # Use weightshared FPN model for image and target
            input_targets = [KL.Lambda(lambda x: x[:,idx,...])(input_target) for idx in range(input_target.shape[1])]
            for k, one_target in enumerate(input_targets):
                T2,T3,T4,T5 = ResNet(101, name='resnet101')(one_target)
                out = FPN(config.TOP_DOWN_PYRAMID_SIZE)([T2,T3,T4,T5])
                if k == 0:
                    target_pyramid = out
                else:
                    target_pyramid = [KL.Add(name=f"target_adding_{k}_{i}")([target_pyramid[i], out[i]]) for i in range(len(target_pyramid))]
            TP2,TP3,TP4,TP5,TP6 = [KL.Lambda(lambda x: x/config.NUM_TARGETS)(target_pyramid[i]) for i in range(len(target_pyramid))]
            P2 = L1_distance(3*config.FPN_FEATUREMAPS//2, name="P2")([P2, TP2])
            P3 = L1_distance(3*config.FPN_FEATUREMAPS//2, name="P3")([P3, TP3])
            P4 = L1_distance(3*config.FPN_FEATUREMAPS//2, name="P4")([P4, TP4])
            P5 = L1_distance(3*config.FPN_FEATUREMAPS//2, name="P5")([P5, TP5])
            P6 = L1_distance(3*config.FPN_FEATUREMAPS//2, name="P6")([P6, TP6])

        
        anchors = build_anchors(config)

        rpn_class_logits, rpn_class, rpn_bbox = RPN(config.RPN_ANCHOR_STRIDE, 
                                                    len(config.RPN_ANCHOR_RATIOS))(rpn_feature_maps)
        # ----------NOTE RCNN+MASK Layers NOTE -----------
        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training" else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(proposal_count=proposal_count,
                                nms_threshold=config.RPN_NMS_THRESHOLD,
                                name="ROI",
                                config=config)([rpn_class, rpn_bbox, anchors])

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            # ************************* NOTE for 2 label dataset 
            active_class_ids = parse_image_meta_graph(input_image_meta, config)["active_class_ids"]
            if config.NUM_CLASSES2 > 2:
                active_class_ids2 = parse_image_meta_graph(input_image_meta, config)["active_class_ids2"]


            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4], name="input_roi", dtype=np.int32)
                # Normalize coordinates
                target_rois = norm_boxes_graph(input_rois, input_image.shape.as_list()[1:3])
            else:
                target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            output_rois, target_class_ids, target_class_ids2, target_bbox, target_mask, target_text_embed, \
                target_embed_length, target_rbox = DetectionTargetLayer(
                    config, name="proposal_targets")([target_rois, input_gt_class_ids, gt_boxes, 
                    input_gt_masks,input_gt_class_ids2, input_gt_text_embed, input_gt_embed_length, input_gt_rboxes])


            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            mrcnn_box_outputs = Classifier(config)([output_rois, input_image_meta, mrcnn_feature_maps])
            mrcnn_mask = Mask(config)([output_rois, input_image_meta, mrcnn_feature_maps])
            
            # Losses
            rpn_class_loss = RPN_CLA_LOSS(config)([input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = RPN_BBOX_LOSS(config)([input_rpn_bbox, input_rpn_match, rpn_bbox])
            if config.MODEL == "mrcnn":
                class_loss = MRCNN_CLA_LOSS(config)([target_class_ids, mrcnn_box_outputs["mrcnn_class_logits"], active_class_ids])
            else:
                # CHANGE: use custom class loss without using active_class_ids
                class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                    [target_class_ids, mrcnn_box_outputs["mrcnn_class_logits"], active_class_ids])
            bbox_loss = MRCNN_BBOX_LOSS(config)([target_bbox, target_class_ids, mrcnn_box_outputs["mrcnn_bbox"]])
            mask_loss = MRCNN_MASK_LOSS(config)([target_mask, target_class_ids, mrcnn_mask])
            losses = {"rpn_class_loss": rpn_class_loss,
                      "rpn_bbox_loss": rpn_bbox_loss,
                      "class_loss": class_loss,
                      "bbox_loss": bbox_loss,
                      "mask_loss": mask_loss}
            # ************************* NOTE for 2 label dataset 
            if config.HAVE_LABEL2:
                class_loss2 = MRCNN_CLA2_LOSS(config)([target_class_ids2, target_class_ids, mrcnn_box_outputs["mrcnn_class_logits2"], active_class_ids2])
                losses["class_loss2"] = class_loss2
            if config.READ:
                ctc_loss = MRCNN_READ_LOSS(config)([input_gt_recognition, input_gt_text_embed, input_gt_text_length, input_gt_embed_length])
                losses["ctc_loss"] = ctc_loss
            if config.RBOX:
                rbbox_loss = MRCNN_RBOX_LOSS(config)([target_rbbox, target_class_ids, mrcnn_rbbox])
                losses["rbbox_loss"] = rbbox_loss
            
            # Model
            # ************************* NOTE for 2 label dataset 
            inputs = [input_image, input_image_meta, input_rpn_match, input_rpn_bbox,
                    input_gt_class_ids, input_gt_class_ids2, input_gt_boxes, input_gt_rboxes, input_gt_masks,
                    input_gt_text_embed, input_gt_embed_length]
            if config.MODEL == "smrcnn":
                inputs = [input_image, input_image_meta, input_target, input_rpn_match, input_rpn_bbox,
                        input_gt_class_ids, input_gt_class_ids2, input_gt_boxes, input_gt_rboxes, input_gt_masks,
                        input_gt_text_embed, input_gt_embed_length]
            # inputs = [input_image, input_image_meta, input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
            
            outputs = [rpn_class_logits, rpn_class, rpn_bbox, mrcnn_mask, rpn_rois, output_rois]\
                 + [v for k,v in mrcnn_box_outputs.items()] + [v for k,v in losses.items()]

            model = KM.Model(inputs, outputs=losses, name='mask_rcnn')
        # evaluate or inference
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_box_outputs = Classifier(config)([rpn_rois, input_image_meta, mrcnn_feature_maps])

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
            # normalized coordinates
            if config.HAVE_LABEL2:
                detections = DetectionLayer(config, name="mrcnn_detection")(
                    [rpn_rois, mrcnn_box_outputs["mrcnn_class_logits"], 
                    mrcnn_box_outputs.get("mrcnn_class_logits2"), 
                    mrcnn_box_outputs["mrcnn_bbox"], input_image_meta])
            else:
                detections = DetectionLayer(config, name="mrcnn_detection")(
                    [rpn_rois, mrcnn_box_outputs["mrcnn_class_logits"], 
                    mrcnn_box_outputs["mrcnn_bbox"], input_image_meta])

            # Create masks for detections
            # detection_boxes = KL.Lambda(lambda x: x[..., :4], name='detections')(detections)
            # ************************* NOTE for 2 label dataset 
            mrcnn_mask = Mask(config)([detections[..., :4], input_image_meta, mrcnn_feature_maps])
            inputs = [input_image, input_image_meta, input_anchors]
            if config.MODEL == "smrcnn":
                inputs = [input_image, input_image_meta, input_target, input_anchors]
            model = KM.Model([input_image, input_image_meta, input_anchors],
                             [detections, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox] + [v for k,v in mrcnn_box_outputs.items()],
                             name='mask_rcnn')        

        
        return model

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
        -----------------------------------------------------------
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                f"Could not find model directory under {self.model_dir}")
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("OOD"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, f"Could not find weight files in {dir_name}")
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        Params:
        -----------------------------------------------------------
            exclude: list of layer names to exclude
        """

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)
            print(f"Exclude layers:\n\t{[l for l in exclude]}")

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def get_imagenet_weights(self):
        """Downloads ImageNet trained weights from Keras.
        Returns:
        -----------------------------------------------------------
            path to weights file.
        """
        from tensorflow.keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/'\
                                 'releases/download/v0.2/'\
                                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        return weights_path

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        if self.config.ADAMW:
            from nets.adamw import AdamW
            optimizer = AdamW(lr=learning_rate, decay=0.001, weight_decay=self.config.WEIGHT_DECAY, 
                clipnorm=self.config.GRADIENT_CLIP_NORM)
        else:
            optimizer = keras.optimizers.SGD(
                lr=learning_rate, momentum=momentum,
                clipnorm=self.config.GRADIENT_CLIP_NORM)
            
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        # ************************* NOTE for 2 label dataset 
        if self.config.HAVE_LABEL2:
            loss_names = [
                "rpn_class_loss",  "rpn_bbox_loss",
                "mrcnn_class_loss", "mrcnn_class_loss2", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
        else:
            loss_names = [
                "rpn_class_loss",  "rpn_bbox_loss",
                "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]

        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (tf.reduce_mean(layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs))
        # print (self.keras_model.metrics_names)
        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (tf.reduce_mean(layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model._metrics_tensors.update({name: loss})
            # self.keras_model._compile_stateful_metrics_tensors.update({name: loss})
        # print ("================",self.keras_model._compile_stateful_metrics_tensors)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            # print("layer.__class__.__name__", layer.__class__.__name__)
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                log(f"{' '*indent}{layer.name:25}    ({layer.__class__.__name__})")
    
    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.
        Params:
        -----------------------------------------------------------
            model_path: If None, or a format different from what this code uses
                then set a new log directory and start epochs from 0. Otherwise,
                extract the log directory and the epoch counter from the file
                name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:

            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]OOD\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, f"{self.config.NAME.lower()}{now:%Y%m%dT%H%M}")

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, f"OOD_{self.config.NAME.lower()}_*epoch*.h5")
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        Params:
        -----------------------------------------------------------
            learning_rate: The learning rate to train with
            epochs:        Number of training epochs. Note that previous training epochs
                           are considered to be done alreay, so this actually determines
                           the epochs to train in total rather than in this particaular
                           call.
            layers: Allows selecting wich layers to train. It can be:
                    - A regular expression to match layer names to train
                    - One of these predefined values:
                    heads: The RPN, classifier and mask heads of the network
                    all: All the layers
                    3+: Train Resnet stage 3 and up
                    4+: Train Resnet stage 4 and up
                    5+: Train Resnet stage 5 and up
            augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
                          augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
                          flips images right/left 50% of the time. You can pass complex
                          augmentations as well. This augmentation applies 50% of the
                          time, and when it does it flips images right/left half the time
                          and adds a Gaussian blur with a random sigma in range 0 to 5.
                          augmentation = imgaug.augmenters.Sometimes(0.5, [
                              imgaug.augmenters.Fliplr(0.5),
                              imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                          ])
            custom_callbacks: Optional. Add custom callbacks to be called
                              with the keras fit_generator method. Must be list of type keras.callbacks.
            no_augmentation_sources: Optional. List of sources to exclude for
                                     augmentation. A source is string that identifies a dataset and is
                                     defined in the Dataset class.
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            # "3+": fr"(res3.*)|({self.config.NORM_TYPE}3.*)|(res4.*)|({self.config.NORM_TYPE}4.*)|(res5.*)|({self.config.NORM_TYPE}5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(nas_fpn\_.*)",
            # "4+": fr"(res4.*)|({self.config.NORM_TYPE}4.*)|(res5.*)|({self.config.NORM_TYPE}5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(nas_fpn\_.*)",
            "4+": fr"(res.*)|(rpn.*)|(mrcnn.*)",
            "5+": fr"(res5.*)|({self.config.NORM_TYPE}5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(nas_fpn\_.*)",
            # From a specific Mobilenet stage and up
            "5M+": r"(mob.*5.*)|(mob.*6.*)|(mob.*7.*)|(mob.*8.*)|(mob.*9.*)|(mob.*10.*)|(mob.*11.*)|(mob.*12.*)|(mob.*13.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(nas_fpn\_.*)",
            "11M+": r"(mob.*11.*)|(mob.*12.*)|(mob.*13.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(nas_fpn\_.*)",
            "13M+": r"(mob.*13.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(nas_fpn\_.*)",
            # From a specifig Mnasnet stage and up
            "7mn+": r"(mnas.*7.*)|(mnas.*8.*)|(mnas.*9.*)|(mnas.*10.*)|(mnas.*11.*)|(mnas.*12.*)|(mnas.*13.*)|(mnas.*14.*)|(mnas.*15.*)|(mnas.*16.*)|(mnas.*17.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(nas_fpn\_.*)",
            "12mn+": r"(mnas.*12.*)|(mnas.*13.*)|(mnas.*14.*)|(mnas.*15.*)|(mnas.*16.*)|(mnas.*17.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(nas_fpn\_.*)",
            "17mn+": r"(mnas.*17.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(nas_fpn\_.*)",
            # From a specifig Xception stage and up
            "3x+": r"(xcept.*block3.*)|(xcept.*block4.*)|(xcept.*block5.*)|(xcept.*block6.*)|(xcept.*block7.*)|(xcept.*block8.*)|(xcept.*block9.*)|(xcept.*block10.*)|(xcept.*block11.*)|(xcept.*block12.*)|(xcept.*block13.*)|(xcept.*block14.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(nas_fpn\_.*)",
            "4x+": r"(xcept.*block4.*)|(xcept.*block5.*)|(xcept.*block6.*)|(xcept.*block7.*)|(xcept.*block8.*)|(xcept.*block9.*)|(xcept.*block10.*)|(xcept.*block11.*)|(xcept.*block12.*)|(xcept.*block13.*)|(xcept.*block14.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(nas_fpn\_.*)",
            "5x+": r"(xcept.*block13.*)|(xcept.*block14.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(nas_fpn\_.*)",
            # From a specifig Nasnet stage and up
            "4n+": r"(nas_6.*)|(nas_7.*)|(nas_8.*)|(nas_9.*)|(nas_10.*)|(nas_11.*)|(nas_12.*)|(nas_13.*)|(nas_14.*)|(nas_15.*)|(nas_16.*)|(nas_17.*)|(nas_18.*)|(nas_last_relu)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(nas_fpn\_.*)",
            "5n+": r"(nas_13.*)|(nas_14.*)|(nas_15.*)|(nas_16.*)|(nas_17.*)|(nas_18.*)|(nas_last_relu)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(nas_fpn\_.*)",

            # All layers
            "all": ".*",
        }

        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE,
                                         no_augmentation_sources=no_augmentation_sources)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE)

        # train_dataset = DataLoader(train_dataset, self.config, augmentation)
        # train_generator = DataGenerator(train_dataset)
        # # print ("next train")
        # # print (train_generator)
        # train_generator = tf.data.Dataset.from_generator(train_generator, 
        #     (tf.float32, tf.float32, tf.int32, tf.int32, tf.float32, tf.float32, tf.bool, tf.bool, tf.float32, tf.int32, tf.int32, tf.int32, tf.float32))
        # train_generator = train_generator.padded_batch(
        #     self.config.BATCH_SIZE, padded_shapes=([None, None, None], [None], [None], [None], [None, 4], [None, 5],
        #     [None, None], [None, None, None], [None], [None, None], [None], [None, 1], [None, 4]))
        # # train_generator = train_generator.make_one_shot_iterator()
        # train_generator = iter(train_generator)
        

        # val_dataset = DataLoader(val_dataset, self.config, augmentation)
        # val_generator = DataGenerator(val_dataset)
        # # print (val_dataset[0])
        # val_generator = tf.data.Dataset.from_generator(val_generator, 
        #     (tf.float32, tf.float32, tf.int32, tf.int32, tf.float32, tf.float32, tf.bool, tf.bool, tf.float32, tf.int32, tf.int32, tf.int32, tf.float32))
        # val_generator = val_generator.padded_batch(
        #     self.config.BATCH_SIZE, padded_shapes=([None, None, None], [None], [None], [None], [None, 4], [None, 5],
        #     [None, None], [None, None, None], [None], [None, None], [None], [None, 1], [None, 4]))
        # # val_generator = val_generator.make_one_shot_iterator()
        # val_generator = iter(val_generator)
        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=0, write_graph=True, write_images=True),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,verbose=0, save_weights_only=True),
        ]


        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        log(f"Starting at epoch {self.epoch}. LR={learning_rate}")
        log(f"Checkpoint Path:\n\t{self.checkpoint_path}")
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers.
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()
            # workers = 8


        # self.keras_model.fit(
        #     train_generator,
        #     initial_epoch=self.epoch,
        #     epochs=epochs,
        #     steps_per_epoch=self.config.STEPS_PER_EPOCH,
        #     callbacks=callbacks,
        #     validation_data=val_generator,
        #     validation_steps=self.config.VALIDATION_STEPS,
        #     max_queue_size=128, # 100
        #     workers=1,
        #     use_multiprocessing=False, #True
        # )
        
        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=128, # 100
            workers=1,
            use_multiprocessing=False, #True
        )
        self.epoch = max(self.epoch, epochs)

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        Params:
        -----------------------------------------------------------
            images: List of image matrices [height,width,depth]. Images can have
                    different sizes.
        Returns:
        -----------------------------------------------------------
        3 Numpy matrices:
            molded_images: [N, h, w, 3]. Images resized and normalized.
            image_metas:   [N, length of meta data]. Details about each image.
            windows:       [N, (y1, x1, y2, x2)]. The portion of the image that has the
                           original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                    0, image.shape, molded_image.shape, window, scale,
                    np.zeros([self.config.NUM_CLASSES], dtype=np.int32),
                    np.zeros([self.config.NUM_CLASSES2], dtype=np.int32))

            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                          image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.
        Params:
        -----------------------------------------------------------
            detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
            mrcnn_mask: [N, height, width, num_classes]
            original_image_shape: [H, W, C] Original image shape before resizing
            image_shape: [H, W, C] Shape of the image after resizing and padding
            window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                    image is excluding the padding.
        Returns:
        -----------------------------------------------------------
            boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
            class_ids: [N] Integer class IDs for each bounding box
            scores: [N] Float probability scores of the class_id
            masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        if self.config.NUM_CLASSES2 > 2:
            boxes = detections[:N, :4]
            class_ids = detections[:N, 4].astype(np.int32)
            scores = detections[:N, 5]
            class_ids2 = detections[:N, 6].astype(np.int32)
            scores2 = detections[:N, 7]
        else:
            # Extract boxes, class_ids, scores, and class-specific masks
            boxes = detections[:N, :4]
            class_ids = detections[:N, 4].astype(np.int32)
            scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            if self.config.NUM_CLASSES2 > 2:
                class_ids2 = np.delete(class_ids2, exclude_ix, axis=0)
                scores2 = np.delete(scores2, exclude_ix, axis=0)
            N = class_ids.shape[0]
        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape, softmask=self.config.SOFT_MASK)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1) if full_masks else np.empty(original_image_shape[:2] + (0,))
        results = {
                "rois": boxes,
                "class_ids": class_ids,
                "scores": scores,
                "masks": full_masks,
            }
        if self.config.NUM_CLASSES2 > 2:
            results.update({
                "class_ids2": class_ids2,
                "scores2": scores2,
            })
        return results

    def detect(self, images, verbose=0):
        """Runs the detection pipeline.
        Params:
        -----------------------------------------------------------
            images: List of images, potentially of different sizes.
        Returns:
        -----------------------------------------------------------
        a list of dicts, one dict per image. The dict contains:
            rois:       [N, (y1, x1, y2, x2)] detection bounding boxes
            class_ids:  [N] int class IDs
            scores:     [N] float probability scores for the class IDs
            class_ids2: [N] int class2 IDs
            scores2:    [N] float probability scores for the class2 IDs
            masks:      [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log(f"Processing {len(images)} images")
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape,\
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas",   image_metas)
            log("anchors",       anchors)
        # Run object detection
        # ************************* NOTE for 2 label dataset 

        predict = self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        detections,mrcnn_mask = predict[:2]
        # Process detections
        results = []
        for i, image in enumerate(images):
            result = self.unmold_detections(detections[i], mrcnn_mask[i],
                                    image.shape, molded_images[i].shape,
                                    windows[i])
            results.append(result)
        return results

    def detect_molded(self, molded_images, image_metas, verbose=0):
        """Runs the detection pipeline, but expect inputs that are
        molded already. Used mostly for debugging and inspecting
        the model.
        Params:
        -----------------------------------------------------------
            molded_images: List of images loaded using load_image_gt()
            image_metas:   image meta data, also returned by load_image_gt()
        Returns:
        -----------------------------------------------------------
        a list of dicts, one dict per image. The dict contains:
            rois:      [N, (y1, x1, y2, x2)] detection bounding boxes
            class_ids: [N] int class IDs
            scores:    [N] float probability scores for the class IDs
            masks:     [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(molded_images) == self.config.BATCH_SIZE, "Number of images must be equal to BATCH_SIZE"

        if verbose:
            log(f"Processing {len(molded_images)} images")
            for image in molded_images:
                log("image", image)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, "Images must have the same size"

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas",   image_metas)
            log("anchors",       anchors)
        # Run object detection
        predict = self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        detections,mrcnn_mask = predict[:2]
        # Process detections
        results = []
        for i, image in enumerate(molded_images):
            window = [0, 0, image.shape[0], image.shape[1]]
            result = self.unmold_detections(detections[i], mrcnn_mask[i],
                                    image.shape, molded_images[i].shape,
                                    window)
            results.append(result)
        return results

