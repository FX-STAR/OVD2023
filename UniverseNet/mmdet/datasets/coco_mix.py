import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .coco import CocoDataset
from numpy import random 


@DATASETS.register_module()
class CocoMixDataset(CocoDataset):
    def __init__(self, ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 mixup=0.0,
                 mosaic=0.0):
        super(CocoMixDataset, self).__init__(ann_file=ann_file,
                 pipeline=pipeline,
                 classes=classes,
                 data_root=data_root,
                 img_prefix=img_prefix,
                 seg_prefix=seg_prefix,
                 proposal_file=proposal_file,
                 test_mode=test_mode,
                 filter_empty_gt=filter_empty_gt)
        assert 0<=mixup<=1, 'mixup overflow'
        assert 0<=mosaic<=1, 'mosaic overflow'
        self.mixup = mixup 
        self.mosaic = mosaic
    
    def prepare_train_img(self, idx):
        if self.mixup==0 and self.mosaic==0:
            idxs = [idx]
        else:
            # 0-0.4: ori, 0.4-0.7:mix, 0.7-1: mosaic
            prob = random.uniform(0, 1)
            if prob<(1-self.mixup-self.mosaic):
                idxs = [idx]
            elif prob<(1-self.mosaic):
                idxs = [idx]+list(random.randint(1, len(self.data_infos)-1, 1))
            else:
                idxs = [idx]+list(random.randint(1, len(self.data_infos)-1, 3))
        imgs_info = [self.data_infos[i] for i in idxs]
        anns_info = [self.get_ann_info(i) for i in idxs]
        results = dict(img_info=imgs_info, ann_info=anns_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)