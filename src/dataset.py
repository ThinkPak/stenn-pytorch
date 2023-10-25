import json
import os
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from zhdate import ZhDate
from datetime import datetime


class SITS_Dataset(data.Dataset):
    def __init__(self, folder, folds=None):
        self.folder = folder
        self.folds = folds
        # 读取地块的配置信息，包括地块ID和分组，根据ID获取分组，再根据分组获取均值和标准差
        self.meta_patch = gpd.read_file(os.path.join(folder, "metadata.geojson"))
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        self.meta_patch.sort_index(inplace=True)
        # 读取指定分组中的样本
        if folds is not None:
            self.meta_patch = pd.concat(
                [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
            )
        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index
        # 读取样本的采集时间
        dates = self.meta_patch["dates-S2"]
        self.date_index = {}
        for id, date_seq in dates.items():
            d = pd.DataFrame().from_dict(date_seq, orient="index")
            d[0] = d[0].apply(
                lambda x: datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
            )
            d[1] = d[0].apply(
                lambda x: ZhDate.from_datetime(x).newyear
            )
            self.date_index[id] = np.array((d[0] - d[1]).apply(lambda x: x.days))
        # 读取规范化所用的值
        with open(os.path.join(folder, "NORM_S2_patch.json"), "r") as f:
            norm_val = json.loads(f.read())
        folds = range(1, 6)
        self.means = [norm_val["Fold_{}".format(f)]["mean"] for f in folds]
        self.stds = [norm_val["Fold_{}".format(f)]["std"] for f in folds]

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        id_patch = self.id_patches[item]
        # 读取卫星影像时间序列，并转成Tensor
        file_path = os.path.join(self.folder, "DATA_S2", "S2_{}.npy".format(id_patch))
        data = np.load(file_path).astype(np.float32)
        data = torch.from_numpy(data)
        # 读取卫星影像对应的均值和标准差
        fold = self.meta_patch["Fold"][int(id_patch)]
        mean = torch.from_numpy(np.array(self.means[fold - 1])).float()
        std = torch.from_numpy(np.array(self.stds[fold - 1])).float()
        # 规范化：使用影像对应的均值和标准差
        data = (data - mean[None, :, None, None]) / std[None, :, None, None]
        # 获取时间
        dates = torch.from_numpy(self.date_index[id_patch].astype(int))
        # 读取卫星影像标签数据，并转成Tensor
        file_path = os.path.join(self.folder, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch))
        target = np.load(file_path)[0].astype(int)
        target = torch.from_numpy(target)
        return (data, dates), target, id_patch
