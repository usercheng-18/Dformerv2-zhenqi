# 注意这里改为了绝对路径导入
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module()
class ACOD12KDataset(CustomDataset):
    """ACOD-12K dataset for Camouflaged Object Detection."""

    # 二分类：背景 和 伪装目标
    CLASSES = ('background', 'camouflaged')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(ACOD12KDataset, self).__init__(
            img_suffix='.jpg',  # 注意：检查你的图片是不是 .jpg，如果不是请修改
            seg_map_suffix='.png',  # GT 掩码通常是 .png
            reduce_zero_label=False,
            **kwargs)