import os
import cv2
import torch
import numpy as np
import torch.utils.data as data


def get_path(
        dataset_name,
        _rgb_path,
        _rgb_format,
        _x_path,
        _x_format,
        _gt_path,
        _gt_format,
        x_modal,
        item_name,
):
    if dataset_name == "ACOD-12K":
        rgb_path = os.path.join(_rgb_path, item_name + _rgb_format)
        gt_path = os.path.join(_gt_path, item_name + _gt_format)
        # 核心逻辑：解决 _left_ 与 _depth_ 的命名差异
        depth_item_name = item_name.replace("_left_", "_depth_")
        d_path = os.path.join(_x_path, depth_item_name + _x_format)

    # ... 其他数据集逻辑保持不变 ...
    elif dataset_name == "StanFord2D3D":
        rgb_path = os.path.join(_rgb_path, item_name.replace(".jpg", "").replace(".png", "") + _rgb_format)
        d_path = os.path.join(_x_path,
                              item_name.replace(".jpg", "").replace(".png", "").replace("/rgb/", "/depth/").replace(
                                  "_rgb", "_newdepth") + _x_format)
        gt_path = os.path.join(_gt_path,
                               item_name.replace(".jpg", "").replace(".png", "").replace("/rgb/", "/semantic/").replace(
                                   "_rgb", "_newsemantic") + _gt_format)
    # ... (此处省略部分重复的 elif) ...
    else:
        clean_name = item_name.split("/")[-1].split(".")[0] if "/" in item_name else item_name.split(".")[0]
        rgb_path = os.path.join(_rgb_path, clean_name + _rgb_format)
        d_path = os.path.join(_x_path, clean_name + _x_format)
        gt_path = os.path.join(_gt_path, clean_name + _gt_format)

    path_result = {"rgb_path": rgb_path, "gt_path": gt_path}
    for modal in x_modal:
        path_result[modal + "_path"] = eval(modal + "_path")
    return path_result


class RGBXDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(RGBXDataset, self).__init__()
        self.setting = setting
        self._split_name = split_name
        self._rgb_path = setting["rgb_root"]
        self._rgb_format = setting["rgb_format"]
        self._gt_path = setting["gt_root"]
        self._gt_format = setting["gt_format"]
        self._transform_gt = setting["transform_gt"]
        self._x_path = setting["x_root"]
        self._x_format = setting["x_format"]
        self._train_source = setting["train_source"]
        self._eval_source = setting["eval_source"]
        self.class_names = setting["class_names"]
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess
        self.dataset_name = setting["dataset_name"]
        self.x_modal = setting.get("x_modal", ["d"])
        self.backbone = setting["backbone"]

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        item_name = self._construct_new_file_names(self._file_length)[index] if self._file_length is not None else \
        self._file_names[index]
        path_dict = get_path(self.dataset_name, self._rgb_path, self._rgb_format, self._x_path, self._x_format,
                             self._gt_path, self._gt_format, self.x_modal, item_name)

        rgb_mode = "RGB" if self.dataset_name == "SUNRGBD" and self.backbone.startswith("DFormerv2") else "BGR"
        rgb = self._open_image(path_dict["rgb_path"], rgb_mode)
        gt = self._open_image(path_dict["gt_path"], cv2.IMREAD_GRAYSCALE, dtype=np.uint8)

        if rgb is None or gt is None:
            h, w = self.setting.get('image_height', 480), self.setting.get('image_width', 640)
            rgb = np.zeros((h, w, 3), dtype=np.uint8) if rgb is None else rgb
            gt = np.zeros((h, w), dtype=np.uint8) if gt is None else gt

        # 核心逻辑：ACOD-12K 的 255 -> 1 映射
        if self._transform_gt:
            gt = self._gt_transform(gt, self.dataset_name)

        x = {}
        for modal in self.x_modal:
            if modal == "d":
                img_d = self._open_image(path_dict[modal + "_path"], cv2.IMREAD_GRAYSCALE)
                img_d = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8) if img_d is None else img_d
                x[modal] = cv2.merge([img_d, img_d, img_d])
            else:
                x[modal] = self._open_image(path_dict[modal + "_path"], "RGB")

        x = x[self.x_modal[0]] if len(self.x_modal) == 1 else x
        if self.preprocess is not None:
            rgb, gt, x = self.preprocess(rgb, gt, x)

        return dict(
            data=torch.from_numpy(np.ascontiguousarray(rgb)).float(),
            label=torch.from_numpy(np.ascontiguousarray(gt)).long(),
            modal_x=torch.from_numpy(np.ascontiguousarray(x)).float(),
            fn=str(path_dict["rgb_path"]),
            n=len(self._file_names)
        )

    def _get_file_names(self, split_name):
        source = self._train_source if split_name == "train" else self._eval_source
        with open(source) as f:
            return [item.strip() for item in f.readlines()]

    def _construct_new_file_names(self, length):
        files_len = len(self._file_names)
        new_file_names = self._file_names * (length // files_len)
        rand_indices = torch.randperm(files_len).tolist()
        new_file_names += [self._file_names[i] for i in rand_indices[: length % files_len]]
        return new_file_names

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        if mode in ["RGB", "BGR", cv2.IMREAD_COLOR]:
            img = cv2.imread(filepath, cv2.IMREAD_COLOR)  # 剥离 Alpha 通道
            if img is not None and mode == "RGB":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(filepath, mode)
        return np.array(img, dtype=dtype) if img is not None else None

    @staticmethod
    def _gt_transform(gt, dataset_name):
        if dataset_name == "ACOD-12K":
            # 解决 CUDA illegal memory access：映射 255 到类索引 1
            new_gt = gt.copy()
            new_gt[gt == 255] = 1
            return new_gt
        return gt - 1

    @classmethod
    def get_class_colors(*args):
        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            id = i
            r, g, b = 0, 0, 0
            for j in range(7):
                r = r ^ (((id >> 0) & 1) << (7 - j))
                g = g ^ (((id >> 1) & 1) << (7 - j))
                b = b ^ (((id >> 2) & 1) << (7 - j))
                id = id >> 3
            cmap[i] = [r, g, b]
        return cmap.tolist()