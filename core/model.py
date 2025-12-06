#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     model.py
@Time     :     2025/11/21 21:17:48
@Author   :     Louis Swift
@Desc     :     
'''

import os 
import copy
import torch 
import colorsys
import numpy as np 
import torch.nn as nn 
from torch import Tensor
from loguru import logger 
import torch.nn.functional as F 
from PIL import Image,ImageDraw,ImageFont


from .enhaner import * 
from .detector import *
from utils.utils import resize_image,get_classes


__all__ = [
    'EnhanceDetectNet'
]

class EnhanceDetectNet(nn.Module):
    def __init__(
            self,
            cfg_enhancer,
            cfg_detector,
            cfg_DSP = None
        ):
        super().__init__()

        #---------------------------------------------------#
        # 1. initialization the Enhancer and Detector
        #---------------------------------------------------#
        self.enhancer = build_enhancer(cfg_enhancer)
        self.detector = build_detector(cfg_detector.model) 

        #---------------------------------------------------#
        # 1.1 load the weight when necessary
        #---------------------------------------------------#
        if cfg_detector.pretrained_weight_path is not None:
            # load the pretrained weight and finetune on the private dataset 
            model_dict      = self.detector.state_dict()
            pretrained_dict = torch.load(cfg_detector.pretrained_weight_path, map_location = 'cpu')
            load_key, no_load_key, temp_dict = [], [], {}
            for k, v in pretrained_dict.items():
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            model_dict.update(temp_dict)
            self.detector.load_state_dict(model_dict)

        #---------------------------------------------------#
        # 2. config about detector
        #---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(cfg_detector.classes_path)
        self.input_shape = cfg_detector.input_shape
        self.confidence  = cfg_detector.confidence
        self.nms_iou     = cfg_detector.nms_iou
        self.letterbox_image = cfg_detector.letterbox_image

        #---------------------------------------------------#
        # 3. decode box in evaluation
        #---------------------------------------------------#
        self.bbox_util = build_dec_bbox(cfg_detector.bbox)

        #---------------------------------------------------#
        # 4. DSP in training
        #---------------------------------------------------#
        if cfg_DSP is not None:
            self.DSP = torch.hub.load(
                repo_or_dir=cfg_DSP.repo_path,
                model=cfg_DSP.name,
                source='local',
                ckpt_path = cfg_DSP.ckpt_path
            )

            # freeze all the parameters in DSP
            for param in self.DSP.parameters():
                param.requires_grad = False
            self.DSP.eval()
        
        #---------------------------------------------------#
        #   5. 画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        logger.info('Complete the Initialization of [EnhanceDetectNet].')

    def forward(
            self,
            x: Tensor,
        ):
        """
            Args:
                x (Union[Tensor,str]):
                    when x is `Tensor`, it goes the training pipeline.
                    when x is `str`, it goes the evaluation pipeline, which means the input `x` is the path of image
        """

        intermediate_images = {}
        # intermediate_images['ori'] = x.clone()
        intermediate_images['damaged'] = x.clone()

        if self.training and hasattr(self,'DSP'):
            
            self.DSP.eval()
            b,c,ori_h,ori_w = x.shape

            style_factor = torch.randn(b,512).to(next(self.parameters()).device)

            x = F.interpolate(
                x,
                size = (512,512),
                mode='bilinear',
                align_corners=False
            ) # bz c 256 256 

            transfered_x = self.DSP(x,style_factor,alpha=0.01)

            x = F.interpolate(
                transfered_x,
                size = (ori_h,ori_w),
                mode='bilinear',
                align_corners=False
            )

            intermediate_images['style'] = x.clone()

        x,gate = self.enhancer(x)

        intermediate_images['enhance'] = x

        outputs = self.detector(x)

        return x, outputs,intermediate_images

    @torch.no_grad()
    def inference(
            self,
            image_path: str,
            bt_mAP: bool,
            intermediate_dir: str
        ):
        """
        Args:
            image_path (str): the path to image.
            bt_mAP (bool): whether to save results in order to calc mAP.
            intermediate_dir (str): the path of directory to save the results.
        """                

        logger.info(f'Start to detect the image [{image_path}].')

        # 1. preprocess the image 
        image = np.asarray(Image.open(image_path).convert('RGB')).astype(np.float32) 
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        image_data = (image_data / 255.).astype(np.float32)
        x = torch.from_numpy(np.transpose(image_data,(2,0,1))).unsqueeze(0).to(next(self.parameters()).device) # BCHW

        # 2. inference
        outputs, intermediate_images = self.forward(x)

        outputs = self.bbox_util.decode_box(outputs)

        #---------------------------------------------------------#
        #   将预测框进行堆叠，然后进行非极大抑制
        #---------------------------------------------------------#
        results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
                    image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)

        if results[0] is None: 
            return 

        # 判断是 YOLOv3（7 列）还是 YOLOv8（6 列）
        data = results[0]
        if data.shape[1] == 7:
            # YOLOv3: x1,y1,x2,y2,obj,cls_prob,cls_id
            top_boxes = data[:, :4]
            top_conf = data[:, 4] * data[:, 5]      # obj × class prob
            top_label = data[:, 6].astype(np.int32)

        elif data.shape[1] == 6:
            # YOLOv8: x1,y1,x2,y2,conf,cls_id
            top_boxes = data[:, :4]
            top_conf = data[:, 4]
            top_label = data[:, 5].astype(np.int32)

        else:
            raise ValueError(f"Unknown bbox format with shape {data.shape}")


        if bt_mAP:
            #---------------------------------------------------------#
            #   保存中间结果，以计算mAP
            #---------------------------------------------------------#
            image_name = image_path.split(os.sep)[-1].split('.')[0]
            with open(intermediate_dir + os.sep + "detection-results/"+image_name+".txt","w") as f:
                for i, c in list(enumerate(top_label)):
                    predicted_class = self.class_names[int(c)]
                    box             = top_boxes[i]
                    score           = str(top_conf[i])

                    top, left, bottom, right = box
                    if predicted_class not in self.class_names:
                        continue

                    f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))
            
            logger.info(f"Saved detection results to [{intermediate_dir}/detection-results/{image_name}.txt]")

        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='configs/simhei.ttf', size=np.floor(3e-2 * image_shape[1] + 0.5).astype('int32'))
        thickness   = int(max((image_shape[0] + image_shape[1]) // np.mean(self.input_shape), 1))
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            # print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image,intermediate_images