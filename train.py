#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     train.py
@Time     :     2025/11/21 12:04:07
@Author   :     Louis Swift
@Desc     :     
'''


import hydra
import torch
from loguru import logger
from omegaconf import Omegaconf
from torch.optim import Adam,SGD
from utils.trainer import Trainer
from utils.logger import setup_logger
from utils.distributed import get_rank
from torch.utils.data import DataLoader
from core.model import EnhanceDetectNet
from core.detector import build_det_loss
from torch.optim.lr_scheduler import MultiStepLR,ExponentialLR
# from utils.graphDataset import GraphDataset, graph_collate_fn
from utils.misc import collect_env,get_exp_info,set_random_seed,get_model_configuration

@logger.catch
@hydra.main(config_path='configs', config_name='config.yaml',version_base=None)
def main(config:Omegaconf):

    #---------------------------------#
    #  print some necessary infomation
    #---------------------------------#
    setup_logger(config.exp.work_dir,get_rank(),f'log_rank{get_rank()}.txt')
    logger.info("Environment info:\n" + collect_env())
    logger.info("Config info:\n" + get_exp_info(config))
    # logger.info("Model Config:\n" + get_model_configuration(config.MODEL_YAML_PATH))

    #---------------------------------#
    #  prepare training
    #---------------------------------#
    set_random_seed(config.exp.random_seed)
    train_dataset = GraphDataset(config,'Train',True)  # Move tensor to the device specified in config.DEVICE
    test_dataset  = GraphDataset(config,'Validation')

    train_loader  = DataLoader(train_dataset,batch_size=config.exp.batch_size,shuffle=True,pin_memory=True,
                               num_workers=config.exp.num_workers,collate_fn=graph_collate_fn,drop_last=True)

    valid_loader   = DataLoader(test_dataset,batch_size=config.exp.batch_size,shuffle=True,pin_memory=True,
                               num_workers=config.exp.num_workers,collate_fn=graph_collate_fn,drop_last=True)
    
    model = EnhanceDetectNet(config.enhancer,config.detector).to(config.DEVICE)
    
    optimizer = Adam(model.parameters(), lr=config.exp.lr,weight_decay=config.exp.weight_decay)
    # lr_scheduler = MultiStepLR(optimizer,milestones=config.MILLESTONES)
    # optimizer = SGD(model.parameters(), lr=config.LR,momentum=config.MOMENTUM,weight_decay=config.WEIGHT_DECAY)
    lr_scheduler = ExponentialLR(optimizer,gamma=config.exp.gamma)

    loss_func = build_det_loss(config.detector.loss,model.detector)

    graphTrainer = Trainer(
        model=model,optimizer=optimizer,lr_scheduler=lr_scheduler,loss_func=loss_func,
        train_loader=train_loader,val_loader=valid_loader,
        **config.exp
    )
    #---------------------------------#
    #  start Training
    #---------------------------------#
    graphTrainer.train()

if __name__ == '__main__':
    main()
    # cProfile.run('main()',filename='TimeAnalysis.out')