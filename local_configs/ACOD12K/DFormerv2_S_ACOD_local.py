import os
import os.path as osp
import time
import numpy as np
from .._base_ import C
from .._base_.datasets.acod12k import *

""" Settings for network """
C.backbone = "DFormerv2_S"
C.pretrained_model = None
C.decoder = "ham"
C.decoder_embed_dim = 512
C.optimizer = "AdamW"

"""Train Config"""
# 4卡加速：单卡 16，总 Batch 为 64
C.batch_size = 16
C.lr = 2.5e-4  # 配合大 Batch Size 调优
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.n_epochs = 500
C.warm_up_epoch = 10

# 重要：四卡并行下的迭代次数计算
num_gpus = 4
C.niters_per_epoch = C.num_train_imgs // (C.batch_size * num_gpus) + 1
C.num_workers = 16 # 利用服务器多核 CPU

C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.drop_path_rate = 0.1
C.aux_rate = 0.0

"""Eval Config"""
C.eval_iter = 10 # 提高验证频率，及时看 mIoU
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]
C.eval_flip = True
C.eval_crop_size = [480, 640]

"""Store Config"""
C.checkpoint_start_epoch = 50
C.checkpoint_step = 50

"""Path Config (保留你的自动化路径逻辑)"""
C.log_dir = osp.abspath("checkpoints/" + C.dataset_name + "_" + C.backbone)
C.log_dir = C.log_dir + "_" + time.strftime("%Y%m%d-%H%M%S", time.localtime()).replace(" ", "_")
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

if not os.path.exists(C.log_dir):
    os.makedirs(C.log_dir, exist_ok=True)
exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
C.log_file = C.log_dir + "/log_" + exp_time + ".log"