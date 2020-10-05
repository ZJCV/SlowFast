# -*- coding: utf-8 -*-

"""
@date: 2020/9/11 下午6:59
@file: lr_scheduler.py
@author: zj
@description: 
"""

import torch.optim as optim
from tsn.config import cfg
from tsn.model.build import build_model
from tsn.optim.build import build_optimizer
from tsn.optim.build import build_lr_scheduler
from tsn.util.checkpoint import CheckPointer
from tsn.util.logger import setup_logger

if __name__ == '__main__':
    cfg.merge_from_file('configs/trn_resnet50_hmdb51_rgb.yaml')

    model = build_model(cfg)
    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)
    # print(model)
    print(optimizer)
    print(lr_scheduler)
    print(lr_scheduler.after_scheduler)
    print(lr_scheduler.state_dict())
    print(lr_scheduler.after_scheduler.state_dict())

    logger = setup_logger(cfg.TRAIN.NAME)
    checkpointer = CheckPointer(model, optimizer=optimizer, scheduler=lr_scheduler, save_dir=cfg.OUTPUT.DIR,
                                save_to_disk=True, logger=logger)

    extra_checkpoint_data = checkpointer.load()

    print(optimizer)
    print(lr_scheduler.optimizer)

    for i in range(1000):
        optimizer.step()
        lr_scheduler.step()
        if i % 100 == 0:
            print(optimizer)
            print(lr_scheduler.optimizer)
