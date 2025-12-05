#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     loss.py
@Time     :     2025/12/05 21:52:29
@Author   :     Louis Swift
@Desc     :     
'''

import torch.nn as nn
from torch import Tensor
from loguru import logger 

from .detector import * 
from .enhaner import * 

class Criterion(nn.Module):
    def __init__(
            self,
            cfg_loss,
            cfg_enhancer,
            cfg_detector,
        ):
        super().__init__()

        self.dip_loss = build_dip_loss(**cfg_enhancer.loss)
        self.det_loss = build_det_loss(**cfg_detector.loss)

        self.lambda_dip = cfg_loss.lambda_dip
        self.lambda_det = cfg_loss.lambda_det

        logger.info('Complete the Initialization of [Loss].')
    def forward(
            self,
            # 1. used for dip 
            pred: Tensor,
            gt: Tensor,
            damaged: Tensor,
            # 2. used for det
            det_output
    ):
        
        # 1. dip loss
        dip_loss ,loss_dict = self.dip_loss(pred, gt, damaged)
        # 2. det_loss
        det_loss = self.det_loss(det_output)
        loss_dict['det'] = det_loss

        # 3. total loss
        loss = self.lambda_dip * dip_loss + self.lambda_det * det_loss

        return loss, loss_dict
