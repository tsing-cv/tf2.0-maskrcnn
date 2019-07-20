import os
import cv2
import numpy as np
import skimage.transform
from pycocotools.coco import COCO
import sys
sys.path.append("../..")
from detection.datasets import transforms, utils
from detection.networks.model import Config
import visualize, show
import matplotlib.pyplot as plt

class CocoDataSet(object):
    def __init__(self, dataset_dir, subset,
                 flip_ratio=0,
                 pad_mode='fixed',
                 config=None,
                 debug=False):
        '''Load a subset of the COCO dataset.
        
        Attributes
        ---
            dataset_dir: The root directory of the COCO dataset.
            subset: What to load (train, val).
            flip_ratio: Float. The ratio of flipping an image and its bounding boxes.
            pad_mode: Which padded method to use (fixed, non-fixed)

            scale: Tuple of two integers.
        '''
        self.config = config
        self.debug = debug
        if subset not in ['train', 'val']:
            raise AssertionError('subset must be "train" or "val".')
            

        self.coco = COCO(f"{dataset_dir}/annotations/instances_{subset}2017.json")

        # get the mapping from original category ids to labels
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        
        self.img_ids, self.img_infos = self._filter_imgs()
        
        if debug:
            self.img_ids, self.img_infos = self.img_ids[:50], self.img_infos[:50]
            
        self.image_dir = f"{dataset_dir}/{subset}2017"
        
        self.flip_ratio = flip_ratio
        
        if pad_mode in ['fixed', 'non-fixed']:
            self.pad_mode = pad_mode
        elif subset == 'train':
            self.pad_mode = 'fixed'
        else:
            self.pad_mode = 'non-fixed'
        
        self.img_transform = transforms.ImageTransform((config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM), 
                                                        config.MEAN_PIXEL, 
                                                        config.STD_PIXEL, pad_mode)
        self.bbox_transform = transforms.BboxTransform()
        
        
    def _filter_imgs(self, min_size=32):
        '''Filter images too small or without ground truths.
        
        Args
        ---
            min_size: the minimal size of the image.
        '''
        # Filter images without ground truths.
        all_img_ids = list(set([_['image_id'] for _ in self.coco.anns.values()]))
        # Filter images too small.
        img_ids = []
        img_infos = []
        for i in all_img_ids:
            info = self.coco.loadImgs(i)[0]
            # img_info {'license': 5, 
            #         'file_name': 'COCO_train2014_000000524286.jpg', 
            #         'coco_url': 'http://images.cocodataset.org/train2014/COCO_train2014_000000524286.jpg', 
            #         'height': 480, 
            #         'width': 640, 
            #         'date_captured': '2013-11-22 01:08:02', 
            #         'flickr_url': 'http://farm4.staticflickr.com/3286/3160643026_c2691d2c55_z.jpg', 
            #         'id': 524286}
            ann_ids = self.coco.getAnnIds(imgIds=i)
            # ann_ids [134791, 166182, 201879, 253370, 279458, 1202801, 1836790]
            ann_info = self.coco.loadAnns(ann_ids)
            # ann_info [{'segmentation': [[154.25, 267.61, 168.92, 278.83, 432.11, 281.42, 461.45, 280.56, 469.21, 267.61]], 
            #            'area': 88909.53885, 
            #            'iscrowd': 0, 
            #            'image_id': 524286, 
            #            'bbox': [137.85, 0.97, 346.03, 280.45], 
            #            'category_id': 73, 
            #            'id': 1099077}, 
            #           ]
            ann = self._parse_ann_info(ann_info)
            
            if min(info['width'], info['height']) >= min_size and ann['labels'].shape[0] != 0:
                img_ids.append(i)
                img_infos.append(info)
        return img_ids, img_infos
        
    def _load_ann_info(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        ann_info = self.coco.loadAnns(ann_ids)
        return ann_info

    def _parse_ann_info(self, ann_info):
        '''Parse bbox annotation.
        
        Args
        ---
            ann_info (list[dict]): Annotation info of an image.
            
        Returns
        ---
            dict: A dict containing the following keys: bboxes, 
                bboxes_ignore, labels.
        '''
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_segments = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            # x1, y1, w, h = ann['bbox']
            x1, y1, w, h = np.array(ann['bbox'], dtype=np.int32)
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [y1, x1, y1 + h - 1, x1 + w - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_segments.append(ann["segmentation"])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            # gt_segments = np.array(gt_segments, dtype=np.float32)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            # gt_segments = np.array([], dtype=np.float32)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        
        ann = dict(bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore, segments=gt_segments)
        return ann
    
    @staticmethod
    def _segments_to_mask(segments, labels, img_shape, softmask=False):
        def mask(ann, img_shape, softmask):
            zeros = np.zeros(img_shape, dtype=np.int32)
            ann = np.array(ann[0], np.int32).reshape([-1, 2])
            mask_init = cv2.fillPoly(zeros, [ann], 1)
            if softmask:
                m = cv2.distanceTransform(mask_init.astype(np.uint8), cv2.DIST_L2, 3)
                m /= np.max(m)
                return m
            else:
                return mask_init
        masks = []
        global_mask = np.zeros(img_shape, dtype=np.float32)
        for ann,class_id in zip(segments, labels):
            m = mask(ann, img_shape, softmask)
            masks.append(m.astype(np.float32))
            global_mask = np.where(m, m*class_id, global_mask)
        masks = np.stack(masks, axis=2)
        return masks, global_mask

    @staticmethod
    def minimize_mask(bbox, mask, mini_shape, softmask=False):
        """Resize masks to a smaller version to reduce memory load.
        Mini-masks can be resized back to image scale using expand_masks()
        See inspect_data.ipynb notebook for more details.
        """
        mask_type = np.float32 if softmask else bool
        mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=mask_type)
        for i in range(mask.shape[-1]):
            # Pick slice and cast to bool in case load_mask() returned wrong dtype
            m = mask[:, :, i].astype(mask_type)
            y1, x1, y2, x2 = bbox[i][:4].astype(np.int32)
            m = m[y1:y2, x1:x2]
            if m.size == 0:
                print("Invalid bounding box with area of zero")
                continue
            # Resize with bilinear interpolation
            m = skimage.transform.resize(m, mini_shape,
                                        order=1, mode="constant", cval=0, clip=True,
                                        preserve_range=False, anti_aliasing=False, 
                                        anti_aliasing_sigma=None)
            if not softmask:
                m = np.around(m).astype(np.bool)
            mini_mask[:, :, i] = m
        return mini_mask

    def __len__(self):
        return len(self.img_infos)
    
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
        img_info = self.img_infos[idx]
        ann_info = self._load_ann_info(idx)
        
        # load the image.
        img = cv2.imread(f"{self.image_dir}/{img_info['file_name']}", cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ori_shape = img.shape
        
        # Load the annotation.
        ann = self._parse_ann_info(ann_info)
        bboxes = ann['bboxes']
        labels = ann['labels']
        segments = ann['segments'] # list
        masks, global_mask = self._segments_to_mask(segments, labels, ori_shape[:2])
        flip = True if np.random.rand() < self.flip_ratio else False
        
        # Handle the image
        img, img_shape, scale_factor, masks, global_mask = self.img_transform(img, masks, global_mask, flip)
        # Resize masks to smaller size to reduce memory usage
        if self.config.USE_MINI_MASK and not self.debug:
            masks = self.minimize_mask(bboxes, masks, self.config.MINI_MASK_SHAPE, softmask=self.config.SOFT_MASK)
        pad_shape = img.shape
        
        # Handle the annotation.
        bboxes, labels = self.bbox_transform(bboxes, labels, img_shape, scale_factor, flip)
        
        # Handle the meta info.
        img_meta_dict = dict({
            'ori_shape': ori_shape,
            'img_shape': img_shape,
            'pad_shape': pad_shape,
            'scale_factor': scale_factor,
            'flip': flip
        })

        img_meta = utils.compose_image_meta(img_meta_dict)
        
        return img, img_meta, bboxes, labels, masks, global_mask

    
    def get_categories(self):
        '''Get list of category names. 
        
        Returns
        ---
            list: A list of category names.
            
        Note that the first item 'bg' means background.
        '''
        return ['bg'] + [self.coco.loadCats(i)[0]["name"] for i in self.cat2label.keys()]

if __name__ == "__main__":
    config = Config()
    img_mean = (123.675, 116.28, 103.53)
    # img_std = (58.395, 57.12, 57.375)
    img_std = (1., 1., 1.)
    train_dataset = CocoDataSet('../../COCO2017/', 'train',
                                flip_ratio=0.5,
                                pad_mode='fixed',
                                config=config,
                                debug=True)
    # for i in range(1000):
    img, img_meta, bboxes, labels, masks, global_mask = train_dataset[1]
    rgb_img = np.round(img + img_mean)
    ori_img = utils.get_original_image(img, img_meta, img_mean)
    imshow = show.visualize_boxes(image=rgb_img.astype(np.uint8),
                                boxes=bboxes,
                                labels=labels,
                                scores=np.ones_like(labels),
                                class_labels=train_dataset.get_categories(),
                                masks=masks.transpose([2,0,1]))
    plt.imshow(imshow)
    plt.show()
        # visualize.display_instances(rgb_img, bboxes, labels, train_dataset.get_categories(), masks=masks)