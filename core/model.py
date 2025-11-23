#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     model.py
@Time     :     2025/11/21 21:17:48
@Author   :     Louis Swift
@Desc     :     
'''


import torch 
import torch.nn as nn 

from .enhaner import * 
from .detector import *

__all__ = [
    'EnhanceDetectNet'
]

class EnhanceDetectNet(nn.Module):
    def __init__(
            self,
            cfg_enhancer,
            cfg_detector
        ):
        super().__init__()

        self.enhancer = build_enhancer(cfg_enhancer)
        self.detector = build_detector(cfg_detector) 

        # self.det_loss = build_det_loss(cfg_detector.loss,self.detector)

        self.bbox_util = build_dec_bbox(cfg_detector.bbox)

    def forward(self, x):
        x = self.enhancer(x)
        outputs = self.detector(x)

        # if not self.training: # eval模式下
        #     outputs = self.bbox_util.decode_box(outputs)

        #     results = self.bbox_util.non_max_suppression(
        #             outputs, self.num_classes, self.input_shape, 
        #     image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
        
        return outputs
