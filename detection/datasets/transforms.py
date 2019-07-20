import numpy as np
from detection.datasets.utils import *

class ImageTransform(object):
    '''Preprocess the image.
    
        1. rescale the image to expected size
        2. normalize the image
        3. flip the image (if needed)
        4. pad the image (if needed)
    '''
    def __init__(self,
                 scale=(800, 1333),
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 pad_mode='fixed'):
        self.scale = scale
        self.mean = mean
        self.std = std
        self.pad_mode = pad_mode

        self.impad_size = max(scale) if pad_mode == 'fixed' else 64

    def __call__(self, img, masks, global_mask=None, flip=False):
        img, scale_factor, new_shape = imrescale(img, self.scale)
        masks = cv2.resize(masks, new_shape, cv2.INTER_NEAREST)
        if global_mask is not None:
            global_mask = cv2.resize(global_mask, new_shape, cv2.INTER_NEAREST)
        img_shape = img.shape
        img = imnormalize(img, self.mean, self.std)

        if flip:
            img = img_flip(img)
            masks = img_flip(masks)
            if global_mask is not None:
                global_mask = img_flip(global_mask)
        if self.pad_mode == 'fixed':
            img = impad_to_square(img, self.impad_size)
            masks = impad_to_square(masks, self.impad_size)
            if len(masks.shape) == 2:
                masks = np.expand_dims(masks, axis=-1)
            if global_mask is not None:
                global_mask = impad_to_square(global_mask, self.impad_size)
                global_mask = np.expand_dims(global_mask, axis=-1)
        else: # 'non-fixed'
            img = impad_to_multiple(img, self.impad_size)
            masks = impad_to_multiple(masks, self.impad_size)
            if global_mask is not None:
                global_mask = impad_to_multiple(global_mask, self.impad_size)
        if global_mask is not None:
            return img, img_shape, scale_factor, masks, global_mask
        else:
            return img, img_shape, scale_factor, masks

class BboxTransform(object):
    '''Preprocess ground truth bboxes.
    
        1. rescale bboxes according to image size
        2. flip bboxes (if needed)
    '''
    def __init__(self):
        pass
    
    def __call__(self, bboxes, labels, 
                 img_shape, scale_factor, flip=False):
 
        bboxes = bboxes * scale_factor
        if flip:
            bboxes = bbox_flip(bboxes, img_shape)
            
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[0])
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[1])
            
        return bboxes, labels
