# -*- coding: utf-8 -*-
#================================================================
#
#   Author      : tsing-cv
#   Created date: 2019-02-20 18:32:26
#
#================================================================
import os
import skimage.io as io
import sys
sys.setrecursionlimit(10000)
import time
import cv2
import scipy.ndimage as scimg
import numpy as np
import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# tf.enable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# KTF.set_session(sess)
# KTF.clear_session()
# please refer https://github.com/aleju/imgaug
import imgaug
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
# about coco, please refer https://github.com/waleedka/coco/tree/master/PythonAPI/pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params
from pycocotools import mask as maskUtils
from collections import defaultdict # just for COCOeval

import zipfile
import urllib.request
import shutil
# Root directory of the project
ROOT_DIR = os.path.abspath("../")

sys.path.append(ROOT_DIR) 
from nets.config import Config
from nets import model as modellib
from nets import utils
from tools import visualize
from tensorflow.keras.utils import plot_model # only for plot model structure


# FIXME arguments-------------------------------------------
# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "OOD_mydata.h5")
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2017"
AUTO_DOWNLOAD = False
# ----------------------------------------------------------



def load_letters():
    with open('dictionary.txt', encoding='utf8') as f1:
        str_word = f1.read()
        list_word = list(str_word)
        list_word = [y for y in list_word if y not in '\', ']
        list_word = list_word[1:-1]
    return list_word

############################################################
#  Configurations
############################################################
class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "mydata" # "coco" or other name( for your own dataset name )
    MODEL = "mrcnn"
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2 # must be > 1
    STEPS_PER_EPOCH = 20
    # Number of classes (including background)
    NUM_CLASSES = 1 + 79  # background + numoflabels
    NUM_CLASSES2 = 1

    
    RPN_ANCHOR_RATIOS = [1/14, 1/10, 1/7, 1/4, 1/2, 1, 2, 4, 7, 10, 14]
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  
    

    
    # 'iou' or 'giou'
    # IOU_TYPE = 'giou'
    # ADAMW = True # NOTE optimizor leads to bad result

    # BACKBONE = "xception" # leads to bad result
    # TRAIN_BN = True
    # 'bn' or 'gn', 'sn'
    # NORM_TYPE = 'sn'
    RPN_NMS_THRESHOLD = 0.7
    LEARNING_RATE = 0.001

    # RPN_CLASS_LOSS_TYPE = 'focal_loss' # NOTE leads to bad result
    RPN_BBOX_LOSS_TYPE = 'shape_gradient_smooth_l1_loss'
    # RCNN_CLASS_LOSS_TYPE = 'focal_loss' # NOTE leads to bad result
    RCNN_BBOX_LOSS_TYPE = 'shape_gradient_smooth_l1_loss'
    # MASK_LOSS_TYPE = 'binary_ce'
    
    READ = False
    # SOFT_MASK = True
    # TWO_HEAD = True
    PAN = False
    # NAS_FPN = True # leads to bad result
    # NOTE contain space letter
    NUM_WORDS = len(load_letters()) + 1
    MASKIOU = False

class MyCOCO(COCO):
    def __init__(self, annotation_file):
        self.cats2 = {}
        super(self.__class__, self).__init__(annotation_file)

    def createIndex(self):
        cats2 = {}
        if 'categories2' in self.dataset:
            for cat in self.dataset['categories2']:
                cats2[cat['id']] = cat
        else:
            cats2 = {0: {'supercategory': '###', 'id': 0, 'name': '###'}}
        self.cats2 = cats2
        super(self.__class__, self).createIndex()

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """Rewrite parent function
        filtering parameters. default skips that filter.
        Params:
        -----------------------------------------------------------
            catNms (str array)  : get cats for given cat names
            supNms (str array)  : get cats for given supercategory names
            catIds (int array)  : get cats for given cat ids
        Return: 
        -----------------------------------------------------------
            ids (int array)   : integer array of cat ids
        """
        ids1 = super(self.__class__, self).getCatIds(catNms, supNms, catIds)
        ids2 = [0]
        if self.dataset.get('categories2', []) != []:
            ids2 = [cat['id'] for cat in self.dataset['categories2']]

        return ids1, ids2
    
    def loadCats2(self, ids=[]):
        """
        Load cats with the specified ids.
        Params:
        -----------------------------------------------------------
            ids (int array)       : integer ids specifying cats
        Returns: 
        -----------------------------------------------------------
            cats (object array) : loaded cat objects
        """
        if type(ids) == list:
            return [self.cats2[id] for id in ids]
        elif type(ids) == int:
            return [self.cats2[ids]]  



class MyCOCOeval(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        Params:
        -----------------------------------------------------------
            cocoGt: coco object with ground truth annotations
            cocoDt: coco object with detection results
        Returns:
        ----------------------------------------------------------- 
            None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.params   = {}                  # evaluation parameters
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds()[0])


############################################################
#  Dataset
############################################################
class CocoDataset(utils.Dataset):
    def __init__(self, config, class_map=None):
        super(self.__class__, self).__init__(class_map)
        self.config = config
        self.active_classes = [] # NOTE for siamese

    def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
                  return_coco=False, auto_download=False):
        """Load a subset of the COCO dataset.
        Params:
        -----------------------------------------------------------
            Dir  dataset_dir:   The root directory of the COCO dataset.
            Str  subset:        What to load (train, val, minival, valminusminival)
            Time year:          What dataset year to load (2014, 2017) as a string, not an integer
            List class_ids:     If provided, only loads images that have the given classes.
            Bool return_coco:   If True, returns the COCO object.
            Bool auto_download: Automatically download and unzip MS-COCO images and annotations
        """
        if auto_download is True:
            self.auto_download(dataset_dir, subset, year)

        coco = MyCOCO(f"{dataset_dir}/annotations/instances_{subset}{year}.json")
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = f"{dataset_dir}/images/{subset}{year}"
        print(f"Images path\n\t{os.path.abspath(image_dir)}")
        print(f"Annotations path\n\t{os.path.abspath(dataset_dir)}/annotations/instances_{subset}{year}.json")


        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids, class_ids2 = coco.getCatIds()
            class_ids = sorted(class_ids)
            class_ids2 = sorted(class_ids2)

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
            
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for class_id in class_ids:
            self.add_class("coco", class_id, coco.loadCats(class_id)[0]["name"])
        for class_id2 in class_ids2:
            self.add_class("coco_label2", class_id2, coco.loadCats2(class_id2)[0]["name"])       

        # Add images
        for image_id in image_ids:
            self.add_image(source      = 'coco', 
                        image_id    = image_id,
                        path        = os.path.join(image_dir, coco.imgs[image_id]['file_name']),
                        width       = coco.imgs[image_id]["width"],
                        height      = coco.imgs[image_id]["height"],
                        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id], catIds=class_ids, iscrowd=None)))
            # NOTE show dataset ----------------------------------------
            # if you want to see the dataset, excute here
            # try:
            #     img = io.imread(coco.loadImgs(image_id)[0]['coco_url'])
            # except:
            #     img = io.imread(f"{image_dir}/{coco.loadImgs(image_id)[0]['file_name']}")
            # plt.figure(figsize=(16, 16))
            # plt.imshow(img); plt.axis('off')
            # plt.title(coco.loadImgs(image_id)[0]['file_name'])
            # coco.showAnns(coco.loadAnns(coco.getAnnIds(imgIds=[image_id], catIds=class_ids, iscrowd=None)))
            # plt.show()
            # ----------------------------------------------------------
        
        if return_coco:
            return coco
        

    
    def auto_download(self, dataDir, dataType, dataYear):
        """Download the COCO dataset/annotations if requested.
        Params:
        -----------------------------------------------------------
            dataDir:  The root directory of the COCO dataset.
            dataType: What to load (train, val, minival, valminusminival)
            dataYear: What dataset year to load (2014, 2017) as a string, not an integer
        Note:
        -----------------------------------------------------------
            For 2014, use "train", "val", "minival", or "valminusminival"
            For 2017, only "train" and "val" annotations are available
        """
        # Setup paths and file names
        if dataType == "minival" or dataType == "valminusminival":
            imgDir     = f"{dataDir}/val{dataYear}"
            imgZipFile = f"{dataDir}/val{dataYear}.zip"
            imgURL     = f"http://images.cocodataset.org/zips/val{dataYear}.zip"
        else:
            imgDir     = f"{dataDir}/{dataType}{dataYear}"
            imgZipFile = f"{dataDir}/{dataType}{dataYear}.zip"
            imgURL     = f"http://images.cocodataset.org/zips/{dataType}{dataYear}.zip"
        # print("Image paths:"); print(imgDir); print(imgZipFile); print(imgURL)

        # Create main folder if it doesn't exist yet
        if not os.path.exists(dataDir):
            os.makedirs(dataDir)

        # Download images if not available locally
        if not os.path.exists(imgDir):
            os.makedirs(imgDir)
            print("Downloading images to " + imgZipFile + " ...")
            with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading.")
            print("Unzipping " + imgZipFile)
            with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
                zip_ref.extractall(dataDir)
            print("... done unzipping")
        print("Will use images in " + imgDir)

        # Setup annotations data paths
        annDir = f"{dataDir}/annotations"
        if dataType == "minival":
            annZipFile = f"{dataDir}/instances_minival2014.json.zip"
            annFile    = f"{annDir}/instances_minival2014.json"
            annURL     = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
            unZipDir   = annDir
        elif dataType == "valminusminival":
            annZipFile = f"{dataDir}/instances_valminusminival2014.json.zip"
            annFile    = f"{annDir}/instances_valminusminival2014.json"
            annURL     = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
            unZipDir   = annDir
        else:
            annZipFile = f"{dataDir}/annotations_trainval{dataYear}.zip"
            annFile    = f"{annDir}/instances_{dataType}{dataYear}.json"
            annURL     = f"http://images.cocodataset.org/annotations/annotations_trainval{dataYear}.zip"
            unZipDir   = dataDir
        # print("Annotations paths:"); print(annDir); print(annFile); print(annZipFile); print(annURL)

        # Download annotations if not available locally
        if not os.path.exists(annDir):
            os.makedirs(annDir)
        if not os.path.exists(annFile):
            if not os.path.exists(annZipFile):
                print("Downloading zipped annotations to " + annZipFile + " ...")
                with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
                    shutil.copyfileobj(resp, out)
                print("... done downloading.")
            print("Unzipping " + annZipFile)
            with zipfile.ZipFile(annZipFile, "r") as zip_ref:
                zip_ref.extractall(unZipDir)
            print("... done unzipping")
        print("Will use annotations in " + annFile)


    @staticmethod
    def text_to_labels(text, letters):
        return list(map(lambda x: letters.index(x), text))

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        
        Returns:
        -----------------------------------------------------------
            masks:     A bool array of shape [height, width, instance count] with
                       one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(self.__class__, self).load_mask(image_id)
        
        global_mask = np.zeros([image_info["height"], image_info["width"]])
        instance_masks = []
        class_ids = []
        class_ids2 = [] # label2
        text_embeds = [] # for ocr
        embed_lengths = []
        # load all word
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(f"coco.{annotation['category_id']}")
            class_id2 = config.NUM_CLASSES #self.map_source_class_id(f"coco_label2.{annotation['category_id2']}")
            text = self.map_source_class_id(f"coco_label2.{annotation['text']}") if self.config.READ else ""
            if class_id:
                if not self.config.SOFT_MASK:
                    m = self.annToMask(annotation, image_info["height"], image_info["width"])
                else:
                    m = self.soft_mask(annotation['segmentation'], image_info["height"], image_info["width"])
                global_mask = np.where(m, m*class_id, global_mask).astype(m.dtype)
                # plt.imshow(m)
                # plt.show()
                # time.sleep(10)
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                # print (np.max(m))
                if np.sum(m) <= 0:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                letters = load_letters()
                text_embed = self.text_to_labels(text, letters)
                instance_masks.append(m)
                class_ids.append(class_id)
                class_ids2.append(class_id2-self.config.NUM_CLASSES)
                text_embeds.append(text_embed)
                embed_lengths.append(len(text_embed))
        
        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            if not self.config.SOFT_MASK:
                mask = mask.astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            class_ids2 = np.array(class_ids2, dtype=np.int32)
            text_embeds = np.array(text_embeds, dtype=np.int32)
            embed_lengths = np.array(embed_lengths, dtype=np.int32)
        else:
            # Call super class to return an empty mask
            mask, class_ids = super(self.__class__, self).load_mask(image_id)
            class_ids2 = np.empty([0], np.int32)
            text_embeds = np.empty([[0]], np.int32)
            embed_lengths = np.empty([0], np.int32)
        
        return global_mask, mask, class_ids, class_ids2, text_embeds, embed_lengths

                
            
    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            # super(self.__class__, self).image_reference(image_id)
            return os.path.abspath(coco.imgs[image_id]['file_name'])


    # The following two functions are from pycocotools with a few changes.
    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        Return: 
        -----------------------------------------------------------
            binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        Return:
        ----------------------------------------------------------- 
            binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    def soft_mask(self, ann, height, width):
        """Convert annotation to soft mask
        """
        zeros = np.zeros([height, width], dtype=np.int16)
        ann = np.array(ann, np.int32).reshape(-1, 1, 2)
        mask_init = cv2.fillPoly(zeros, [ann], 1)
        ## method 1
        # mask_i = mask_sum = mask_init
        # i = 0
        # while np.sum(mask_i) > 0:
        #     # if i%10 == 0:
        #     #     print (i)
        #     #     plt.imshow(mask_i)
        #     #     plt.show()
        #     element = np.ones((3,3), dtype=mask_i.dtype)
        #     mask_i = cv2.erode(mask_i, element, iterations=1)
        #     mask_sum += mask_i
        #     i += 1
        # m = np.divide(mask_sum, i)
        ## method 2
        m = cv2.distanceTransform(mask_init.astype(np.uint8), cv2.DIST_L2, 3)
        m /= np.max(m)
        return m

    # NOTE for siamese
    def set_active_classes(self, active_classes):
        """active_classes could be an array of integers (class ids), or
           a filename (string) containing these class ids (one number per line)"""
        if type(active_classes) == str:
            with open(active_classes, 'r') as f:
                content = f.readlines()
            active_classes = [int(x.strip()) for x in content]
        self.active_classes = list(active_classes)
        
    def get_class_ids(self, active_classes, dataset_dir, subset, year):
        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        class_ids = sorted(list(filter(lambda c: c in coco.getCatIds(), self.active_classes)))
        print ("class_ids", class_ids)
        return class_ids

        self.class_ids_with_holes = class_ids
    
    def build_indices(self):

        self.image_category_index = self._build_image_category_index()
        self.category_image_index = self._build_category_image_index(self.image_category_index)

    def _build_image_category_index(self):

        image_category_index = []
        for im in range(len(self.image_info)):
            # List all classes in an image
            coco_class_ids = list(np.unique([self.image_info[im]['annotations'][i]['category_id']\
                                             for i in range(len(self.image_info[im]['annotations']))]\
                                           ))
            # Map 91 class IDs 81 to Mask-RCNN model type IDs
            class_ids = [self.map_source_class_id(f"coco.{coco_class_ids[k]}")\
                         for k in range(len(coco_class_ids))]
            # Put list together
            image_category_index.append(class_ids)

        return image_category_index

    def _build_category_image_index(self, image_category_index):

        category_image_index = []
        # Loop through all 81 Mask-RCNN classes/categories
        for category in range(max(image_category_index)[0]+1):
            # Find all images corresponding to the selected class/category 
            images_per_category = np.where(\
                [any(image_category_index[i][j] == category\
                 for j in range(len(image_category_index[i])))\
                 for i in range(len(image_category_index))])[0]
            # Put list together
            category_image_index.append(images_per_category)

        return category_image_index


############################################################
#  Show pipeline
############################################################
def get_ax(rows=1, cols=1, size=16):
    """返回一个在该notebook中用于所有可视化的Matplotlib Axes array。
    提供一个中央点坐标来控制graph的尺寸。
    
    调整attribute的尺寸来控制渲染多大的图像
    Params:
    -----------------------------------------------------------
        rows: subset rows number
        cols: subset cols number
        size: figure size
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def show_rpn_predict(model, image, gt_class_id, gt_bbox, limit=400):
    """Show rpn processing, draw filtered anchors
    Params:
    -----------------------------------------------------------
        Class   model:       mrcnn model
        Ndarray image:       image
        Int     gt_class_id: class id of groundtruth
        List    gt_bbox:     bbox of groundtruth
    """
    # 生成RPN trainig targets
    # target_rpn_match=1是positive anchors, -1是negative anchors
    # 0是neutral anchors.
    target_rpn_match, target_rpn_bbox = modellib.build_rpn_targets(
        image.shape, model.anchors, gt_class_id, gt_bbox, model.config)
    modellib.log("target_rpn_match", target_rpn_match)
    modellib.log("target_rpn_bbox", target_rpn_bbox)
    
    positive_anchor_ix = np.where(target_rpn_match[:] == 1)[0]
    negative_anchor_ix = np.where(target_rpn_match[:] == -1)[0]
    neutral_anchor_ix = np.where(target_rpn_match[:] == 0)[0]
    positive_anchors = model.anchors[positive_anchor_ix]
    negative_anchors = model.anchors[negative_anchor_ix]
    neutral_anchors = model.anchors[neutral_anchor_ix]
    modellib.log("positive_anchors", positive_anchors)
    modellib.log("negative_anchors", negative_anchors)
    modellib.log("neutral anchors", neutral_anchors)
    
    #将refinement deltas应用于positive anchors
    refined_anchors = utils.apply_box_deltas(
        positive_anchors,
        target_rpn_bbox[:positive_anchors.shape[0]] * model.config.RPN_BBOX_STD_DEV)
    modellib.log("refined_anchors", refined_anchors, )
    
    #显示refinement (点)之前的positive anchors和refinement (线)之后的positive anchors.
    ax = get_ax(1, 2)
    image_with_boxes = visualize.draw_boxes(image, boxes=positive_anchors, 
                         refined_boxes=refined_anchors, 
                         title="Positive and Refined anchors", ax=ax[0])
    plt.imshow(image_with_boxes)
    plt.title("image with boxes")
    plt.show()
    pillar = model.keras_model.get_layer("ROI").output  # node to start searching from

    # TF 1.4 and 1.9 introduce new versions of NMS. Search for all names to support TF 1.3~1.10
    nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression:0")
    if nms_node is None:
        nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV2:0")
    if nms_node is None: #TF 1.9-1.10
        nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV3:0")

    rpn = model.run_graph([image], [
        ("rpn_class", model.keras_model.get_layer("rpn_class").output),
        ("pre_nms_anchors", model.ancestor(pillar, "ROI/pre_nms_anchors:0")),
        ("refined_anchors", model.ancestor(pillar, "ROI/refined_anchors:0")),
        ("refined_anchors_clipped", model.ancestor(pillar, "ROI/refined_anchors_clipped:0")),
        ("post_nms_anchor_ix", nms_node),
        ("proposals", model.keras_model.get_layer("ROI").output),
    ])

    # Show top anchors by score (before refinement)
    sorted_anchor_ids = np.argsort(rpn['rpn_class'][:,:,1].flatten())[::-1]
    visualize.draw_boxes(image, boxes=model.anchors[sorted_anchor_ids[:limit]], title=f"Top {limit} anchors", ax=ax[1])
    plt.show()
    # Show top anchors with refinement. Then with clipping to image boundaries
    ax = get_ax(2, 2)
    pre_nms_anchors = utils.denorm_boxes(rpn["pre_nms_anchors"][0], image.shape[:2])
    refined_anchors = utils.denorm_boxes(rpn["refined_anchors"][0], image.shape[:2])
    refined_anchors_clipped = utils.denorm_boxes(rpn["refined_anchors_clipped"][0], image.shape[:2])
    anchor_limit = 0.5*limit if len(pre_nms_anchors)>0.5*limit else len(pre_nms_anchors)
    visualize.draw_boxes(image, boxes=pre_nms_anchors[:anchor_limit],
                        refined_boxes=refined_anchors[:anchor_limit],
                        title=f"Top {anchor_limit} anchors and refined anchors",
                        ax=ax[0, 0])
    visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[:anchor_limit], title=f"Top {anchor_limit} refined anchors", ax=ax[0, 1])

    # Show refined anchors after non-max suppression
    ixs = rpn["post_nms_anchor_ix"][:anchor_limit]
    visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[ixs], title=f"Top {limit} refined anchors after nms", ax=ax[1, 0])

    # Show final proposals
    # These are the same as the previous step (refined anchors 
    # after NMS) but with coordinates normalized to [0, 1] range.
    # Convert back to image coordinates for display
    h, w = config.IMAGE_SHAPE[:2]
    proposals = rpn['proposals'][0, :limit] * np.array([h, w, h, w])
    visualize.draw_boxes(image, refined_boxes=proposals, title=f"Top {limit} proposals", ax=ax[1, 1])
    plt.show()
    # Measure the RPN recall (percent of objects covered by anchors)
    # Here we measure recall for 3 different methods:
    # - All anchors
    # - All refined anchors
    # - Refined anchors after NMS
    iou_threshold = 0.7

    recall, positive_anchor_ids = utils.compute_recall(model.anchors, gt_bbox, iou_threshold)
    print(f"All Anchors ({model.anchors.shape[0]:5})      Recall: {recall:.3f}  Positive anchors: {len(positive_anchor_ids)}")

    recall, positive_anchor_ids = utils.compute_recall(rpn['refined_anchors'][0], gt_bbox, iou_threshold)
    print(f"Refined Anchors ({rpn['refined_anchors'].shape[1]:5})   Recall: {recall:.3f}  Positive anchors: {len(positive_anchor_ids)}")

    recall, positive_anchor_ids = utils.compute_recall(proposals, gt_bbox, iou_threshold)
    print(f"Post NMS Anchors ({proposals.shape[0]:5})  Recall: {recall:.3f}  Positive anchors: {len(positive_anchor_ids)}")


def show_rois_refinement(model, dataset, config, image, limit=200):
    """Show rcnn predict
    Params:
    -----------------------------------------------------------
        Class   model:   mrcnn model
        Class   dataset: 
        Class   config:  
        Ndarray image:
    """
    # Get input and output to classifier and mask heads.
    mrcnn = model.run_graph([image], [
        ("proposals", model.keras_model.get_layer("ROI").output),
        ("probs", model.keras_model.get_layer("mrcnn_class").output),
        ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),
        ("masks", model.keras_model.get_layer("mrcnn_mask").output),
        ("detections", model.keras_model.get_layer("mrcnn_detection").output),
    ])


    # Get detection class IDs. Trim zero padding.
    det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
    det_count = np.where(det_class_ids == 0)[0][0]
    det_class_ids = det_class_ids[:det_count]
    detections = mrcnn['detections'][0, :det_count]

    print(f"{det_count} detections: {np.array(dataset.class_names)[det_class_ids]}")

    captions = [f"{dataset.class_names[int(c)]} {s:.3f}" if c > 0 else ""
                for c, s in zip(detections[:, 4], detections[:, 5])]
    visualize.draw_boxes(image, 
                        refined_boxes=utils.denorm_boxes(detections[:, :4], image.shape[:2]),
                        visibilities=[2] * len(detections),
                        captions=captions, title="Detections",
                        ax=get_ax())
    
    
    # Proposals的坐标是规范化的坐标. 将它们缩放到图像坐标.
    h, w = config.IMAGE_SHAPE[:2]
    proposals = np.around(mrcnn["proposals"][0] * np.array([h, w, h, w])).astype(np.int32)
    
    # 每个proposal的Class ID, score, and mask
    roi_class_ids = np.argmax(mrcnn["probs"][0], axis=1)
    roi_scores = mrcnn["probs"][0, np.arange(roi_class_ids.shape[0]), roi_class_ids]
    roi_class_names = np.array(dataset.class_names)[roi_class_ids]
    roi_positive_ixs = np.where(roi_class_ids > 0)[0]
    
    #有多少ROIs和空行?
    print(f"{np.sum(np.any(proposals, axis=1))} Valid proposals out of {proposals.shape[0]}")
    print(f"{len(roi_positive_ixs)} Positive ROIs")
    
    # Class数量
    print(list(zip(*np.unique(roi_class_names, return_counts=True))))
    
    #显示一个随机样本的proposals.
    #分类为背景的Proposals是点，其他的显示它们的类名和置信分数.
    ixs = np.random.randint(0, proposals.shape[0], limit)
    captions = [f"{dataset.class_names[c]} {s:.3f}" if c > 0 else ""
                for c, s in zip(roi_class_ids[ixs], roi_scores[ixs])]
    ax = get_ax(1, 2)
    visualize.draw_boxes(image, boxes=proposals[ixs],
                        visibilities=np.where(roi_class_ids[ixs] > 0, 2, 1),
                        captions=captions, title="ROIs Before Refinement",
                        ax=ax[0])

    #指定类别的bounding box偏移.
    roi_bbox_specific = mrcnn["deltas"][0, np.arange(proposals.shape[0]), roi_class_ids]
    modellib.log("roi_bbox_specific", roi_bbox_specific)
    
    #应用bounding box变换
    #形状: [N, (y1, x1, y2, x2)]
    refined_proposals = utils.apply_box_deltas(
        proposals, roi_bbox_specific * config.BBOX_STD_DEV).astype(np.int32)
    modellib.log("refined_proposals", refined_proposals)
    
    #显示positive proposals
    # ids = np.arange(roi_boxes.shape[0])  #显示所有
    if len(roi_positive_ixs) > 5:
        ids = np.random.randint(0, len(roi_positive_ixs), 5)  #随机显示样本
    else:
        ids = roi_positive_ixs
    captions = [f"{dataset.class_names[c]} {s:.3f}" if c > 0 else ""
                for c, s in zip(roi_class_ids[roi_positive_ixs][ids], roi_scores[roi_positive_ixs][ids])]
    visualize.draw_boxes(image, boxes=proposals[roi_positive_ixs][ids],
                        refined_boxes=refined_proposals[roi_positive_ixs][ids],
                        visibilities=np.where(roi_class_ids[roi_positive_ixs][ids] > 0, 1, 0),
                        captions=captions, title="ROIs After Refinement",
                        ax=ax[1])

    #去掉那些被分类为背景的boxes
    keep = np.where(roi_class_ids > 0)[0]
    print(f"Keep {keep.shape[0]} detections:\n{keep}")
    
    #去掉低置信度的检测结果
    keep = np.intersect1d(keep, np.where(roi_scores >= model.config.DETECTION_MIN_CONFIDENCE)[0])
    print(f"Remove boxes below {config.DETECTION_MIN_CONFIDENCE} confidence. Keep {keep.shape[0]}:\n{keep}")
    #为每一个类别做NMS
    pre_nms_boxes = refined_proposals[keep]
    pre_nms_scores = roi_scores[keep]
    pre_nms_class_ids = roi_class_ids[keep]
    
    nms_keep = []
    for class_id in np.unique(pre_nms_class_ids):
        #选择该类的检测结果
        ixs = np.where(pre_nms_class_ids == class_id)[0]
        #做NMS
        class_keep = utils.non_max_suppression(pre_nms_boxes[ixs], 
                                                pre_nms_scores[ixs],
                                                config.DETECTION_NMS_THRESHOLD)
        #映射索引
        class_keep = keep[ixs[class_keep]]
        nms_keep = np.union1d(nms_keep, class_keep)
        print(f"{dataset.class_names[class_id][:20]:22}: {keep[ixs]} -> {class_keep}")
    
    keep = np.intersect1d(keep, nms_keep).astype(np.int32)
    print(f"\nKept after per-class NMS: {keep.shape[0]}\n{keep}")
    
    #显示最终的检测结果
    ixs = np.arange(len(keep))  # Display all
    # ixs = np.random.randint(0, len(keep), 10)  # Display random sample
    captions = [f"{dataset.class_names[c]} {s:.3f}" if c > 0 else ""
                for c, s in zip(roi_class_ids[keep][ixs], roi_scores[keep][ixs])]
    visualize.draw_boxes(
        image, boxes=proposals[keep][ixs],
        refined_boxes=refined_proposals[keep][ixs],
        visibilities=np.where(roi_class_ids[keep][ixs] > 0, 1, 0),
        captions=captions, title="Detections after NMS",
        ax=get_ax())


def show_activate_layers(model, config, image):
    # TODO
    #获取一些示例层的activations
    activations = model.run_graph([image], [
        ("input_image",        model.keras_model.get_layer("input_image").output),
        ("res4w_out",          model.keras_model.get_layer("res4w_out").output),  # for resnet100
        ("rpn_bbox",           model.keras_model.get_layer("rpn_bbox").output),
        ("roi",                model.keras_model.get_layer("ROI").output),
    ])
    
    #输入图像 (规范化的)
    _ = plt.imshow(modellib.unmold_image(activations["input_image"][0],config))
    
    # Backbone feature map
    visualize.display_images(np.transpose(activations["res4w_out"][0,:,:,:4], [2, 0, 1]))



############################################################
#  COCO Evaluation
############################################################
def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Evaluate coco dataset results
    
    Params:
    -----------------------------------------------------------
        Class-Child dataset:   inhere follow the Datset class
        List        image_ids: images id list
        List        rois:      [[xmin, ymin, xmax, ymax]]
        List        class_ids: list of labels
        List        scores:    evaluate confidence
        Array       masks:     masks ndarray 
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score    = scores[i]
            bbox     = np.around(rois[i], 1)
            mask     = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                # "category_id2": dataset.get_source_class_id(class_ids2[i], "coco_label2"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
                }
            results.append(result)
    return results

def build_coco_results2(dataset, image_ids, rois, class_ids, class_ids2, scores, scores2, masks):
    """Evaluate coco dataset results
    
    Params:
    -----------------------------------------------------------
        Class-Child dataset:    inhere follow the Datset class
        List        image_imasksds:  images id list
        List        rois:  masks     [[xmin, ymin, xmax, ymax]]
        List        class_imasksds:  list of labels
        List        scores:masks     evaluate confidence
        List        class_imasksds2: list of labels
        List        scores2masks:    evaluate confidence
        Array       masks: masks     masks ndarray 
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id  = class_ids[i]
            score     = scores[i]
            class_id2 = class_ids2[i]
            score2    = scores2[i]
            bbox      = np.around(rois[i], 1)
            mask      = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "category_id2": dataset.get_source_class_id(class_ids2[i]+8, "coco_label2"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "score2": score2,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
                }
            results.append(result)
    return results

def evaluate_coco(model, dataset, coco, config, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    Params:
    -----------------------------------------------------------
        dataset:   A Dataset object with valiadtion data
        eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
        limit:     if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    APs = []
    IOU_SHRESHOLD = 0.7
    if config.HAVE_LABEL2:
        for i, image_id in enumerate(image_ids):
            # Load image
            # image = dataset.load_image(image_id)
            image, image_meta, gt_class_id, gt_class_id2, gt_bbox, gt_rbox, gt_global_mask, gt_mask, gt_mask_score, gt_text_embeds, gt_embed_lengths =\
                modellib.load_image_gt(dataset, config, image_id)
            print (gt_mask.shape)
            info = dataset.image_info[image_id]
            print(f"image ID: {info['source']}.{info['id']}({image_id}) {dataset.image_info[image_id]['path']}")
            # print (dataset.class_names)
            # visualize.display_images(np.transpose(gt_mask, [2, 0, 1]), titles='gt_mask', cmap="Blues")
            gt_instance = visualize.display_instances_duclass(image     = image, 
                                                            boxes        = gt_bbox, 
                                                            masks        = gt_mask, 
                                                            class_ids    = gt_class_id,
                                                            class_names  = dataset.class_names,
                                                            class_ids2   = gt_class_id2+config.NUM_CLASSES,
                                                            class_names2 = dataset.class_names,
                                                            title        = "Gt instance",
                                                            auto_show    = True,
                                                            soft_mask    = config.SOFT_MASK)

            # Run detection
            t_s = time.time()
            r = model.detect([image], verbose=0)[0]
            t_e = time.time()
            print (t_e-t_s)
            t_prediction += (t_e - t_s)
            # Show instance
            visualize.display_instances_duclass(image       = image, 
                                        boxes        = r['rois'], 
                                        masks        = r['masks'], 
                                        class_ids    = r["class_ids"],
                                        class_names  = dataset.class_names,
                                        scores       = r['scores'],
                                        class_ids2   = r["class_ids2"]+config.NUM_CLASSES,
                                        class_names2 = dataset.class_names,
                                        scores2      = r['scores2'],
                                        title        = "Predict instance",
                                        show_mask    = True,
                                        soft_mask    = config.SOFT_MASK)

            # show rpn and rois procession
            # show_rpn_predict(model=model, image=image, gt_class_id=gt_class_id, gt_bbox=gt_bbox)
            # show_rois_refinement(model=model, dataset=dataset, config=config, image=image)
            # show_activate_layers(model=model, config=config, image=image)
            #画出precision-recall的曲线
            AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                    r['rois'], r['class_ids'], r['scores'], r['masks'],
                                                    IOU_SHRESHOLD, True, gt_class_id2, 
                                                    r['class_ids2'], r['scores2'])

            # visualize.plot_precision_recall(AP, precisions, recalls)
            # # # 显示confusion matrix
            # visualize.plot_overlaps(gt_class_id, r['class_ids'], r['scores'],
            #                         overlaps, dataset.class_names)
            
            # Convert results to COCO format
            # Cast masks to uint8 because COCO tools errors out on bool
            image_results = build_coco_results2(dataset, coco_image_ids[i:i + 1],
                                            r["rois"], r["class_ids"], r["class_ids2"],
                                            r["scores"], r['scores2'],
                                            r["masks"].astype(np.uint8))
            results.extend(image_results)
            APs.append(AP)

        print(f"mAP @ IoU={IOU_SHRESHOLD}: {np.mean(APs)}")

        # Load results. This modifies results with additional attributes.
        coco_results = coco.loadRes(results)

        # # Evaluate
        cocoEval = MyCOCOeval(coco, coco_results, eval_type)
        cocoEval.params.imgIds = coco_image_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        print(f"\nPrediction time: {t_prediction:0.6}. Average waste {t_prediction / len(image_ids):0.6}s/image")
        print(f"Total time: {time.time() - t_start:0.6}")
    
    else:
        for i, image_id in enumerate(image_ids):
            # Load image
            # image = dataset.load_image(image_id)

            image, image_meta, gt_class_id, gt_class_id2, gt_bbox, gt_rbox, gt_global_mask, gt_mask, gt_mask_score, gt_text_embeds, gt_embed_lengths = modellib.load_image_gt(dataset, config, image_id)
            info = dataset.image_info[image_id]
            print(f"image ID: {info['source']}.{info['id']}({image_id}) {dataset.image_info[image_id]['path']}")
            
            # visualize.display_images(np.transpose(gt_mask, [2, 0, 1]), titles='gt_mask', cmap="Blues")
            # gt_instance = visualize.display_instances(image     = image, 
            #                                         boxes       = gt_bbox, 
            #                                         masks       = gt_mask, 
            #                                         class_ids   = gt_class_id,
            #                                         class_names = dataset.class_names,
            #                                         title       = "Gt instance",
            #                                         auto_show   = True,
            #                                         soft_mask   = config.SOFT_MASK)
            # plt.imsave('../gt_instance/{}.jpg'.format(image_id), gt_instance)
            # Run detection
            t_s = time.time()
            with tf.device("/gpu:0"):
                r = model.detect([image], verbose=0)[0]
            t_e = time.time()
            print (t_e-t_s)
            t_prediction += (t_e - t_s)
            # Show instance
            # print (r['masks'])
            visualize.display_instances(image       = image, 
                                        boxes       = r['rois'], 
                                        masks       = r['masks'], 
                                        class_ids   = r["class_ids"],
                                        class_names = dataset.class_names,
                                        scores      = r['scores'],
                                        title       = "Predict instance",
                                        show_mask   = True,
                                        soft_mask   = config.SOFT_MASK)

            # show rpn and rois procession
            # show_rpn_predict(model=model, image=image, gt_class_id=gt_class_id, gt_bbox=gt_bbox)
            # show_rois_refinement(model=model, dataset=dataset, config=config, image=image)
            # show_activate_layers(model=model, config=config, image=image)
            #画出precision-recall的曲线
            # AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
            #                                         r['rois'], r['class_ids'], r['scores'], r['masks'])
            # visualize.plot_precision_recall(AP, precisions, recalls)
            # # 显示confusion matrix
            # visualize.plot_overlaps(gt_class_id, r['class_ids'], r['scores'],
            #                         overlaps, dataset.class_names)
            
            # Convert results to COCO format
            # Cast masks to uint8 because COCO tools errors out on bool
        #     image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
        #                                     r["rois"], r["class_ids"],
        #                                     r["scores"],
        #                                     r["masks"].astype(np.uint8))
        #     results.extend(image_results)
        #     APs.append(AP)

        # print("mAP @ IoU=50: ", np.mean(APs))
        # # Load results. This modifies results with additional attributes.
        # coco_results = coco.loadRes(results)

        # # Evaluate
        # cocoEval = MyCOCOeval(coco, coco_results, eval_type)
        # cocoEval.params.imgIds = coco_image_ids
        # cocoEval.evaluate()
        # cocoEval.accumulate()
        # cocoEval.summarize()

        # print(f"\nPrediction time: {t_prediction:0.6}. Average waste {t_prediction / len(image_ids):0.6}s/image")
        # print(f"Total time: {time.time() - t_start:0.6}")



# *************************************************************************
# -------------------------------------------------------------------------
#                             Control
# -------------------------------------------------------------------------
# *************************************************************************
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train model on MS COCO.')
    parser.add_argument("command", 
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=False,
                        default="../Coco",
                        metavar="coco path",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--model', required=False,
                        default="last",
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--img_path', required=False,
                        default="ant+hill_1_3.jpg",
                        metavar="<image path>",
                        help='Image to use for inference (default=None)')
    args = parser.parse_args()
    print("\n\n---------------------Step1: Command Line Parameters--------------------------")
    print("Command:    \t", args.command)
    print("Model:      \t", os.path.abspath(args.model))
    print("Dataset:    \t", os.path.abspath(args.dataset))
    print("Year:       \t", DEFAULT_DATASET_YEAR)
    print("Logs:       \t", DEFAULT_LOGS_DIR)

    
    
    print("\n\n-------------------------Step2: Config Parameters----------------------------")
    # Configurations
    if args.command == "train":
        config = CocoConfig()
    else:
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0 # 0
            DETECTION_MAX_INSTANCES = 200
        config = InferenceConfig()
    config.display()
    

    
    print("\n\n-----------------------Step3: Load Graph and Weghts--------------------------")
    # Create model
    mode = "training" if args.command == "train" else "inference"
    model = modellib.OOD(mode=mode, config=config,
                              model_dir=DEFAULT_LOGS_DIR)
    
    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
        if not os.path.exists(model_path):
            utils.download_trained_weights(COCO_MODEL_PATH)
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model
    
    
    print(f"Model_path\n\t{model_path}")
    exclude=["conv1", 'fpn_c5p5', 'fpn_c4p4', 'fpn_c3p3', 'fpn_c2p2',
            "mrcnn_class_logits", "mrcnn_class_logits2", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask", "rpn_model", "mrcnn_mask_bn1", 
            "mrcnn_mask_bn2", "mrcnn_mask_bn3", "mrcnn_mask_bn4", "mrcnn_class_bn1", 
            "mrcnn_class_bn2", "mrcnn_bbox_logit", "mrcnn_bbox_conv1"] if config.NAME != \
                'coco' and args.command == "train" else None
    model.load_weights(model_path, by_name=True, exclude=exclude)#["resnet101"])
    # print (model.log_dir)
    # 显示所有训练的weights 
    # print("Display Weight Statiscal Graph") 
    # visualize.display_weight_stats(model)
    # Draw model structure
    if args.command == 'train' and os.path.exists(model.log_dir):
        config.display(save_config=True, save_path=f"{model.log_dir}/configuration.yml")
        print(f"Drawing model structure\n\tSaved in {os.path.abspath(f'{model.log_dir}/model.png')}")
    plot_model(model.keras_model,to_file=f'model.png')

    
    print("\n\n-------------------Step4: Execute Train, Eval or Infer-----------------------")
    # Train or evaluate
    if args.command == "train":
        # Train dataset
        dataset_train = CocoDataset(config)
        dataset_train.load_coco(args.dataset, "train", year=DEFAULT_DATASET_YEAR, auto_download=AUTO_DOWNLOAD)
        if DEFAULT_DATASET_YEAR in '2014':
            dataset_train.load_coco(args.dataset, "valminusminival", 
                year=DEFAULT_DATASET_YEAR, auto_download=AUTO_DOWNLOAD)
        dataset_train.prepare()
        # NOTE for siamese
        dataset_train.build_indices()
        dataset_train.ACTIVE_CLASSES = np.array(range(1,config.NUM_CLASSES+1))
        # Validation dataset
        dataset_val = CocoDataset(config)
        val_type = "val" if DEFAULT_DATASET_YEAR in '2017' else "minival"
        dataset_val.load_coco(args.dataset, val_type, year=DEFAULT_DATASET_YEAR, auto_download=AUTO_DOWNLOAD)
        dataset_val.prepare()
        dataset_val.build_indices()
        dataset_val.ACTIVE_CLASSES = np.array(range(1,config.NUM_CLASSES+1))
        # Image Augmentation
        sometimes = lambda aug: imgaug.augmenters.Sometimes(0.5, aug)
        augmentation = imgaug.augmenters.Sequential([#imgaug.augmenters.SomeOf((0, 3),[
            # imgaug.augmenters.Fliplr(0.5),
            # imgaug.augmenters.Flipud(0.5),
            imgaug.augmenters.GaussianBlur(sigma=(0, 0.1)),
            imgaug.augmenters.Add((-20, 20), per_channel=0.5),
            # imgaug.augmenters.Sharpen(alpha=0.4),
            imgaug.augmenters.Grayscale(alpha=(0.0, 0.5)),
            imgaug.augmenters.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5),
            # imgaug.augmenters.ContrastNormalization((0.75, 1.5),per_channel=True),
            imgaug.augmenters.Dropout(p=(0, 0.05)),
            # imgaug.augmenters.Rot90((1, 3)),
            # imgaug.augmenters.Crop(percent=(0, 0.05), keep_size=True),
            imgaug.augmenters.Pad(percent=(0, 0.1), keep_size=True),
            imgaug.augmenters.Affine(scale=(1.1, 1.25),rotate=(-15.,15.), shear=(-8, 8)),
            # imgaug.augmenters.PerspectiveTransform(scale=(0.00, 0.05)),
            # imgaug.augmenters.Multiply((0.8, 1.2), per_channel=0.2),
            imgaug.augmenters.ChangeColorspace(to_colorspace='BGR'),
            sometimes(
                imgaug.augmenters.ElasticTransformation(alpha=(0.0, 0.5), sigma=0.25)
            )],
            random_order=True
        )

        # Restore dataset class names for inference
        with open('dataset_class_names.txt', 'w') as f:
            for class_name in dataset_train.class_names:
                f.write(f'{class_name}\n')

        # NOTE This training schedule is an example. Update to your needs NOTE
        # Training - Stage 1
        # print("\nStep1: Training network heads")
        # ops = model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=20,
        #             layers='heads',
        #             augmentation=augmentation)
        

        # # Training - Stage 2
        # # Finetune layers from ResNet stage 4 and up
        # if config.BACKBONE == "mobilenet224v1":
        #     stage_2_layers = '11M+'
        # elif config.BACKBONE == "mnasnet":
        #     stage_2_layers = '12mn+'
        # elif config.BACKBONE == "xception":
        #     stage_2_layers = '4x+'
        # elif config.BACKBONE == "nasnet":
        #     stage_2_layers = '4n+'
        # else:
        #     stage_2_layers = '4+'

        # print(f"Step2: Fine tune {config.BACKBONE} stage {stage_2_layers} and up")        
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=60,
        #             layers=stage_2_layers,
        #             augmentation=augmentation)
                    

        # Training - Stage 3
        # Fine tune all layers
        print("Step3: Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=400,
                    layers='all',
                    augmentation=augmentation)

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = CocoDataset(config)
        val_type = "val" if DEFAULT_DATASET_YEAR in '2017' else "minival"
        coco = dataset_val.load_coco(args.dataset, val_type, 
            year=DEFAULT_DATASET_YEAR, return_coco=True, auto_download=AUTO_DOWNLOAD)
        dataset_val.prepare()
        print(f"Running COCO evaluation on {args.limit} images.")
        evaluate_coco(model, dataset_val, coco, config, "bbox", limit=int(args.limit))

    elif args.command == 'inference':
        import glob
        image_paths = glob.glob(f"/home/aikaka/wqq/OOD/Coco/images/train2017/*.jpg")
        for img_path in image_paths:
            image = plt.imread(img_path)
            dataset_class_names = []
            with open('dataset_class_names.txt', 'r') as f:
                for class_name in f:
                    dataset_class_names.append(class_name.strip()) 
            r = model.detect([image], verbose=0)[0]
            if config.HAVE_LABEL2:
                visualize.display_instances_duclass(image        = image, 
                                                    boxes        = r['rois'], 
                                                    masks        = r['masks'], 
                                                    class_ids    = r["class_ids"],
                                                    class_names  = dataset_class_names,
                                                    scores       = r['scores'],
                                                    class_ids2   = r["class_ids2"]+config.NUM_CLASSES,
                                                    class_names2 = dataset_class_names,
                                                    scores2      = r['scores2'],
                                                    title        = "Predict instance",
                                                    show_mask    = False,
                                                    soft_mask    = config.SOFT_MASK)
            else:
                visualize.display_instances(image        = image, 
                                            boxes        = r['rois'], 
                                            masks        = r['masks'], 
                                            class_ids    = r["class_ids"],
                                            class_names  = dataset_class_names,
                                            scores       = r['scores'],
                                            title        = "Predict instance",
                                            show_mask    = False,
                                            soft_mask    = True)
        # image = plt.imread(args.img_path)
        # dataset_class_names = []
        # with open('dataset_class_names.txt', 'r') as f:
        #     for class_name in f:
        #         dataset_class_names.append(class_name.strip()) 
        # r = model.detect([image], verbose=0)[0]
        # if config.HAVE_LABEL2:
        #     visualize.display_instances_duclass(image        = image, 
        #                                         boxes        = r['rois'], 
        #                                         masks        = r['masks'], 
        #                                         class_ids    = r["class_ids"],
        #                                         class_names  = dataset_class_names,
        #                                         scores       = r['scores'],
        #                                         class_ids2   = r["class_ids2"]+config.NUM_CLASSES,
        #                                         class_names2 = dataset_class_names,
        #                                         scores2      = r['scores2'],
        #                                         title        = "Predict instance",
        #                                         show_mask    = False,
        #                                         soft_mask    = config.SOFT_MASK)
        # else:
        #     visualize.display_instances(image        = image, 
        #                                 boxes        = r['rois'], 
        #                                 masks        = r['masks'], 
        #                                 class_ids    = r["class_ids"],
        #                                 class_names  = dataset_class_names,
        #                                 scores       = r['scores'],
        #                                 title        = "Predict instance",
        #                                 show_mask    = False,
        #                                 soft_mask    = True)
    else:
        print(f"'{args.command}' is not recognized. Use 'train' , 'evaluate' or 'inference'")
