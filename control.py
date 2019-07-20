# !/usr/bin/env python3.7
# -*- encoding: utf-8 -*-
# ***************************************************
# @File    :   train.py
# @Time    :   2019/06/27 11:54:27
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
import tensorflow as tf
import numpy as np
import visualize, show
import matplotlib.pyplot as plt
from tqdm import tqdm
from detection.datasets import coco, data_generator, utils
from detection.networks.model import Model, Config, BoxTransforms, parse_image_meta
# eager execution
tf.enable_eager_execution()
tf.executing_eagerly()

class Helmsman():
    def __init__(self, mode="train", debug=False):
        class TrainConfig(Config):
            NUM_CLASSES = 1 + 80
            IMAGES_PER_GPU = 2
            DETECTION_MIN_CONFIDENCE = 0.
            STEPS_PER_EPOCH = 500
        config = TrainConfig()
        if mode != "train":
            class InferenceConfig(TrainConfig):
                IMAGES_PER_GPU = 1
            config = InferenceConfig()
        config.display()
        
        self.mode = mode
        self.device(config)
        self.dataset(config, debug=debug)
        self.summary_writer = tf.contrib.summary.create_file_writer(config.MODEL_DIR)
        
        assert mode in ["train", "evaluate", "predict"], f"mode must be one of ['train','evaluate','predict']"
        if mode == "train":
            self.train(config=config)
        elif mode == "evaluate":
            self.evaluate(config=config)
        else:
            self.predict(config=config)
    def device(self, config):
        # tensorflow config - using one gpu and extending the GPU 
        # memory region needed by the TensorFlow process
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        sess_config.gpu_options.allow_growth = True
        session = tf.Session(config=sess_config)
    
    def dataset(self, config, debug=False):
        # #### load dataset
        train_dataset = coco.CocoDataSet('./COCO2017', 'train',
                                        flip_ratio=0.,
                                        pad_mode='fixed',
                                        config=config,
                                        debug=False)
        print (train_dataset.get_categories())
        assert config.NUM_CLASSES == len(train_dataset.get_categories()), f"NUM_CLASSES must be compare with dataset, set:{config.NUM_CLASSES} != {len(train_dataset.get_categories())}"                 
        train_generator = data_generator.DataGenerator(train_dataset)        
        train_tf_dataset = tf.data.Dataset.from_generator(
            train_generator, (tf.float32, tf.float32, tf.float32, tf.int32, tf.float32, tf.float32))
        self.train_tf_dataset = train_tf_dataset.padded_batch(
            config.IMAGES_PER_GPU, padded_shapes=([None, None, None], # img
                                                  [None], #img_meta
                                                  [None, None], #bboxes
                                                  [None], #labels
                                                  [None, None, None], #masks 
                                                  [None, None, 1])) # global_mask
        eval_dataset = coco.CocoDataSet('./COCO2017', 'val',
                                        flip_ratio=0.,
                                        pad_mode='fixed',
                                        config=config,
                                        debug=False)
        eval_generator = data_generator.DataGenerator(eval_dataset)        
        eval_tf_dataset = tf.data.Dataset.from_generator(
            eval_generator, (tf.float32, tf.float32, tf.float32, tf.int32, tf.float32, tf.float32))
        self.eval_tf_dataset = eval_tf_dataset.padded_batch(
            config.IMAGES_PER_GPU, padded_shapes=([None, None, None], # img
                                                  [None], #img_meta
                                                  [None, None], #bboxes
                                                  [None], #labels
                                                  [None, None, None], #masks 
                                                  [None, None, 1])) # global_mask
        if debug:
            idx = np.random.choice(range(len(train_dataset)))
            img, img_meta, bboxes, labels, masks, global_mask = train_dataset[idx]
            rgb_img = np.round(img + config.MEAN_PIXEL)
            ori_img = utils.get_original_image(img, img_meta, config.MEAN_PIXEL)
            visualize.display_instances(rgb_img, bboxes, labels, train_dataset.get_categories())
        self.train_dataset = train_dataset
    
    # @tf.function
    def train_step(self, inputs, model, optimizer):
        with tf.GradientTape() as tape:
            loss_dict = model(inputs, training=True)
            loss_value = tf.add_n([i for i in loss_dict.values()])
            loss_dict["total_loss"] = loss_value
        grads = tape.gradient(loss_value, model.variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())
        return loss_dict
    
    def predict_step(self, inputs, model):
        img, img_meta = inputs
        imgs = tf.Variable(np.expand_dims(img, 0))
        img_metas = tf.Variable(np.expand_dims(img_meta, 0))
        detections_list, masks = model([imgs, img_metas], training=False)
        return model.unmold_detections(detections_list, masks, img_metas)[0]

    def train(self, config):

        # #### load model
        model = Model(config)
        # #### overfit a sample
        optimizer = tf.train.MomentumOptimizer(1e-3, 0.9, use_nesterov=False)
        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, directory=config.MODEL_DIR, max_to_keep=5)
        start_epoch = 0
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
            print (f'Latest checkpoint restored!!\n\tModel path is {ckpt_manager.latest_checkpoint}')

        iterator = tf.data.make_one_shot_iterator(self.train_tf_dataset)
        eval_iterator = tf.data.make_one_shot_iterator(self.eval_tf_dataset)
        epochs = 100000
        for epoch in range(start_epoch, epochs):
            loss_history = []
            print ("Training ...")
            for step,inputs in enumerate(iterator):
                start = time.time()
                # plt.imshow(inputs[0][0].numpy())
                # plt.show()
                loss_dict = self.train_step(inputs, model, optimizer)
                end = time.time()
                print(f"""TEP@{epoch:04d}${step:05d} LOSS >> {loss_dict['total_loss'].numpy():.06f} RPN >> {loss_dict['rpn_class_loss'].numpy():.06f}cls {loss_dict['rpn_bbox_loss'].numpy():.06f}reg MRN >> {loss_dict['mrcnn_class_loss'].numpy():.06f}cls {loss_dict['mrcnn_bbox_loss'].numpy():.06f}reg {loss_dict['mrcnn_mask_loss'].numpy():.06f}msk""")
                loss_history.append(loss_dict["total_loss"].numpy())
                # tf.summary.scalar('Train/rpn_class_loss', loss_dict["rpn_class_loss"])
                # tf.summary.scalar('Train/rpn_bbox_loss', loss_dict["rpn_bbox_loss"])
                # tf.summary.scalar('Train/mrcnn_class_loss', loss_dict["mrcnn_class_loss"])
                # tf.summary.scalar('Train/mrcnn_bbox_loss', loss_dict["mrcnn_bbox_loss"])
                # tf.summary.scalar('Train/mrcnn_mask_loss', loss_dict["mrcnn_mask_loss"])
                # tf.summary.scalar('Train/total_loss', loss_dict["total_loss"])
                if step == 500:#config.STEPS_PER_EPOCH-1:
                    break

            print ("Evaluating ...")
            # for step,inputs in enumerate(eval_iterator):
            #     pass

            if epoch%100 == 0:
                ckpt_manager.save()

    def predict(self, path=None, config=None):
        # restore model
        model = Model(config)
        ckpt = tf.train.Checkpoint(model=model)
        ckpt_manager = tf.train.CheckpointManager(ckpt, directory=config.MODEL_DIR, max_to_keep=5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print (f'Latest checkpoint restored!!\n\tModel path is {ckpt_manager.latest_checkpoint}')
        
        idx = np.random.choice(range(len(self.train_dataset)))
        print (f"Predicting image id: {idx}")
        img, img_meta, bboxes, labels, masks, global_mask = self.train_dataset[idx]
        res = self.predict_step([img, img_meta], model)
        print ("res",res)
        ori_img = utils.get_original_image(img, img_meta, config.MEAN_PIXEL)
        # visualize.display_instances(ori_img, res['rois'], res['class_ids'], 
        #                     self.train_dataset.get_categories(), res['masks'], scores=res['scores'])
        imshow = show.visualize_boxes(image=ori_img.astype(np.uint8),
                                    boxes=res['rois'].astype(np.int32),
                                    labels=res['class_ids'],
                                    scores=res['scores'],
                                    class_labels=self.train_dataset.get_categories(),
                                    #masks=res['masks'].transpose([2,0,1])
                                    )
        plt.title(f"{idx}")
        plt.imshow(imshow)
        plt.show()
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train model on MS COCO.')
    parser.add_argument("command", 
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--debug', required=False,
                        default=False,
                        metavar="debug or not",
                        help='If debug, dataset will be show')
    args = parser.parse_args()
    Helmsman(mode=args.command, debug=args.debug)


