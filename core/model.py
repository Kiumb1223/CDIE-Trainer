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
import torch.nn.functional as F 
from typing import Union,Optional
from PIL import Image,ImageDraw,ImageFont

from .enhaner import * 
from .detector import *

__all__ = [
    'EnhanceDetectNet'
]


#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


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
                ckpt_dict = cfg_DSP
            )

            # frezon all the parameters in DSP
            for param in self.DSP.parameters():
                param.requires_grad = False
            self.DSP.eval()
        
        #---------------------------------------------------#
        #   5. 画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    def forward(
            self,
            x: Union[Tensor,str],
            # intermediate_dir:Optional[str]=None
        ):
        """

        Args:
            x (Union[Tensor,str]):
                when x is `Tensor`, it goes the training pipeline.
                when x is `str`, it goes the evaluation pipeline, which means the input `x` is the path of image
            ~~intermediate_dir (Optional[str], optional): the intermedia directory to save some important data. Defaults to None.~~

        """

        intermediate_images = {}

        if self.training and hasattr(self,'DSP'):
            self.DSP.eval()
            b,c,ori_h,ori_w = x.shape
            style_factor = torch.randn(b,512).to(next(self.parameters()).device)
            x = F.interpolate(
                x,
                size = (256,256),
                mode='bilinear',
                align_corners=False
            ) # bz c 256 256 
            transfered_x = self.DSP(x,style_factor)
            x = F.interpolate(
                transfered_x,
                size = (ori_h,ori_w),
                mode='bilinear',
                align_corners=False
            )
            intermediate_images['style'] = x.clone()
        else: # eval mode
            assert isinstance(x,str)
            # intermediate_dir = '.' if intermediate_dir is None else intermediate_dir
            # os.makedirs(intermediate_dir)
            image_path = copy.deepcopy(x) 
            image = np.asarray(Image.open(image_path).convert('RGB'))
            image_shape = np.array(np.shape(image)[0:2])
            #---------------------------------------------------------#
            #   给图像增加灰条，实现不失真的resize
            #   也可以直接resize进行识别
            #---------------------------------------------------------#
            image_data = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
            image_data = (image_data / 255.).astype(np.float32)
            x = torch.from_numpy(np.transpose(image_data,(2,0,1))).unsqueeze(0).to(next(self.parameters()).device) # BCHW

        x,gate = self.enhancer(x)

        intermediate_images['enhance'] = x.clone()

        outputs = self.detector(x)

        if self.training: # train mode 
            return outputs,intermediate_images

        else: # eval mode 
            
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
            
            """
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
            """

            #---------------------------------------------------------#
            #   图像绘制
            #---------------------------------------------------------#
            font        = ImageFont.truetype(font='configs/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
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
                print(label, top, left, bottom, right)
                
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
    

    def get_mAP(self,x:Tensor):
        assert NotImplementedError('waiting to implement')