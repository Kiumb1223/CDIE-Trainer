#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     __init__.py
@Time     :     2025/11/21 20:35:24
@Author   :     Louis Swift
@Desc     :     


    Here are some relavent repository about yolo: 

        YoloV3               https://github.com/bubbliiiing/yolo3-pytorch  
        Efficientnet-Yolo3   https://github.com/bubbliiiing/efficientnet-yolo3-pytorch  
        YoloV4               https://github.com/bubbliiiing/yolov4-pytorch
        YoloV4-tiny          https://github.com/bubbliiiing/yolov4-tiny-pytorch
        Mobilenet-Yolov4     https://github.com/bubbliiiing/mobilenet-yolov4-pytorch
        YoloV5-V5.0          https://github.com/bubbliiiing/yolov5-pytorch
        YoloV5-V6.1          https://github.com/bubbliiiing/yolov5-v6.1-pytorch
        YoloX                https://github.com/bubbliiiing/yolox-pytorch
        YoloV7               https://github.com/bubbliiiing/yolov7-pytorch
        YoloV7-tiny          https://github.com/bubbliiiing/yolov7-tiny-pytorch


    Thanks a lot for their brilliant works~
'''

from omegaconf import OmegaConf

from .yolov3 import * 
from .yolov8 import *


__all__ =  [
    'build_detector',
    'build_dec_bbox',
    'build_det_loss',
]

def build_detector(cfg_detector:OmegaConf):

    if cfg_detector.version == 'v8':
        return Yolov8(**cfg_detector)
    elif cfg_detector.version == 'v3':
        return Yolov3(**cfg_detector)
    else:
        raise NotImplementedError(
             f"Detector version '{cfg_detector.version}' is not supported. "
        )
    
def build_dec_bbox(cfg_bbox:OmegaConf):
    if cfg_bbox.version == 'v8':
        return V8DecodeBox(**cfg_bbox)
    elif cfg_bbox.version == 'v3':
        anchors,num_anchors = get_anchors(cfg_bbox.anchors_path)
        return V3DecodeBox(anchors=anchors,**cfg_bbox)
    else:
        raise NotImplementedError(
            f"Decode bbox version '{cfg_bbox.version}' is not supported. "
        )


def build_det_loss(cfg_loss:OmegaConf,model=None):

    if cfg_loss.version == 'v8':
        assert model is not None 
        return V8Loss(**cfg_loss,stride = model.stride)
    elif cfg_loss.version == 'v3':
        anchors,num_anchors = get_anchors(cfg_loss.anchors_path)
        return V3Loss(anchors=anchors,**cfg_loss)
    else:
        raise NotImplementedError(
            f"Loss version '{cfg_loss.version}' is not supported. "
        )