#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     mAP.py
@Time     :     2025/11/25 14:27:35
@Author   :     Louis Swift
@Desc     :     
                Extracted from https://github.com/bubbliiiing/yolov8-pytorch, with minor modification
                Thanks for their brilliant works.

                整个map计算流程，包括获得预测结果、获得真实框、计算VOC_map。

                Recall和Precision不像AP是一个面积的概念，因此在门限值（Confidence）不同时，网络的Recall和Precision值是不同的。
                默认情况下，本代码计算的Recall和Precision代表的是当门限值（Confidence）为0.5时，所对应的Recall和Precision值。

                受到mAP计算原理的限制，网络在计算mAP时需要获得近乎所有的预测框，这样才可以计算不同门限条件下的Recall和Precision值
                因此，本代码获得的map_out/detection-results/里面的txt的框的数量一般会比直接predict多一些，目的是列出所有可能的预测框，
                
'''
import os
os.environ["HYDRA_FULL_ERROR"] = "1"

import hydra 
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from loguru import logger 
import xml.etree.ElementTree as ET

from utils.utils import get_classes
from utils.utils_mAP import get_map
from core.model import EnhanceDetectNet

@logger.catch 
@hydra.main(config_path="config", config_name="mAP",version_base=None)
def main(config:OmegaConf):

    image_ids = open(os.path.join(config.mAP.VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(config.mAP.map_out_path):
        os.makedirs(config.mAP.map_out_path)
    if not os.path.exists(os.path.join(config.mAP.map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(config.mAP.map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(config.mAP.map_out_path, 'detection-results')):
        os.makedirs(os.path.join(config.mAP.map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(config.mAP.map_out_path, 'images-optional')):
        os.makedirs(os.path.join(config.mAP.map_out_path, 'images-optional'))

    class_names, _ = get_classes(config.mAP.classes_path)

    logger.info("Load model.")
    
    config.detector.confidence = config.mAP.confidence 
    config.detector.nms_iou = config.mAP.nms_iou 
    
    model = EnhanceDetectNet(config.enhancer,config.detector)
    logger.info("Load model done.")

    logger.info("Get predict result.")
    for image_id in tqdm(image_ids):
        image_path  = os.path.join(config.mAP.VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
        image       = Image.open(image_path)
        if config.mAP.map_vis:
            image.save(os.path.join(config.mAP.map_out_path, "images-optional/" + image_id + ".jpg"))
        model.inference(image_path,bt_mAP=True,intermediate_dir=config.mAP.map_out_path)
    logger.info("Get predict result done.")
        
    logger.info("Get ground truth result.")
    for image_id in tqdm(image_ids):
        with open(os.path.join(config.mAP.map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
            root = ET.parse(os.path.join(config.mAP.VOCdevkit_path, "VOC2007/Annotations/"+image_id+".xml")).getroot()
            for obj in root.findall('object'):
                difficult_flag = False
                if obj.find('difficult')!=None:
                    difficult = obj.find('difficult').text
                    if int(difficult)==1:
                        difficult_flag = True
                obj_name = obj.find('name').text
                if obj_name not in class_names:
                    continue
                bndbox  = obj.find('bndbox')
                left    = bndbox.find('xmin').text
                top     = bndbox.find('ymin').text
                right   = bndbox.find('xmax').text
                bottom  = bndbox.find('ymax').text

                if difficult_flag:
                    new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                else:
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

    logger.info("Get ground truth result done.")

    logger.info("Get mAP.")
    get_map(config.mAP.MINOVERLAP, True, score_threhold = config.mAP.score_threhold, path = config.mAP.map_out_path)
    logger.info("Get mAP done.")


if __name__ == "__main__":
    main()