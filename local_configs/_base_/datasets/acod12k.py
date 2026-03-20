from .. import C
import os.path as osp
import numpy as np

"""Dataset Path"""
C.dataset_name = "ACOD-12K"
# C.root_dir 在 _base_/__init__.py 中应设为 "/mnt/data1/zhangmq/datasets"
C.dataset_path = osp.join(C.root_dir, "ACOD-12K")
C.rgb_root_folder = osp.join(C.dataset_path, "RGB")
C.rgb_format = ".png"
C.gt_root_folder = osp.join(C.dataset_path, "Label")
C.gt_format = ".png"

# 开启转换，以触发 RGBXDataset.py 中的 255 -> 1 逻辑
C.gt_transform = True

C.x_root_folder = osp.join(C.dataset_path, "Depth")
C.x_format = ".png"
C.x_is_single_channel = True

# 路径指向你随项目一同传输到服务器的 data 文件夹
C.train_source = "/mnt/data1/zhangmq/DFormer-main/data/train.txt"
C.eval_source = "/mnt/data1/zhangmq/DFormer-main/data/test.txt"
C.is_test = True

# 训练参数
C.num_train_imgs = 4600
C.num_eval_imgs = 1492
C.num_classes = 2
C.class_names = ["background", "crop"]

"""Image Config"""
C.background = 255
C.image_height = 480
C.image_width = 640
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

# 提升数据读取效率，利用服务器多核 CPU
C.num_workers = 16