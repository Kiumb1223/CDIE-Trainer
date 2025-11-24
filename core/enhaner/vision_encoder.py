#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     vision_encoder.py
@Time     :     2025/11/24 20:56:20
@Author   :     Louis Swift
@Desc     :     
'''

import torch 
import torch.nn as nn 

__all__ = [
    'VisionEncoder'
]

class VisionEncoder(nn.Module):
    def __init__(
            self,
            num_layers:int = 5,
            base_channel:int = 64,
            encoder_output_dim:int = 256,

            **kwargs
        ):
        super().__init__()

        self.num_layers = num_layers
        self.encoder_output_dim = encoder_output_dim
        self.base_channel = base_channel

        self.convs    = nn.ModuleList()
        self.maxpools = nn.ModuleList()
        self.adpools  = nn.ModuleList()
        self.linears  = nn.ModuleList()

        in_channel = 3 
        out_channel = base_channel

        for i in range(num_layers):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1),
                    nn.ReLU(True)
                )
            )
            if i != num_layers - 1: 
                self.maxpools.append(
                    nn.AvgPool2d((3,3),(2,2))
                )

            self.adpools.append(
                nn.AdaptiveAvgPool2d((1,1))
            )
            self.linears.append(
                nn.Sequential(
                    nn.Linear(out_channel, encoder_output_dim),
                    nn.ReLU(True)
                )
            )
            #---------------------------------#
            # From GDIP-YOLO: 
            #       The number of channels in each layer is double the previous, 
            #       starting from 64 in the first layer and 1024 in the final layer.
            #---------------------------------#
            in_channel = out_channel
            out_channel = out_channel * 2 

    def forward(self,x):
        outputs = []

        for i in range(self.num_layers):
            x = self.convs[i](x)
            if i != self.num_layers - 1:
                x = self.maxpools[i](x)
            else:
                x = x 
            x_adp = self.adpools[i](x)
            x_vec = x_adp.view(x_adp.shape[0], -1)

            outputs.append(self.linears[i](x_vec)) 
        
        return outputs
    
