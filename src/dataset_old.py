import json
import os
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata


class PASTIS_Dataset(tdata.Dataset):
    def __init__(
            self,
            folder,
            norm=True,
            target="semantic",
            cache=False,
            mem16=False,
            folds=None,
            reference_date="2018-09-01",
            class_mapping=None,
            mono_date=None,
            sats=["S2"],
    ):
        """
        使用Pytorch的Dataset类从PASTIS数据集中加载样本，用于语义和全景分割
        Dataset会产生((data, dates), target)元组，其中：
            -data包含图像时间序列
            -dates包含以参考日期以来的天数
            -target是语义或实例目标
        参数：
            folder（str）：数据集的路径
            norm（bool）：如果为真，则使用预先计算的通道均值和标准偏对图像进行标准化。
            target（str）：'semantic'或'instance'。定义DataLoader返回的目标类型。
                若为‘semantic’，则目标张量是包含每个像素类别的张量。
            cache（bool）：若为‘True’，则将样本加载到RAM中，默认为‘False’。
            mem16（bool）：缓存的附加参数。若为‘True’，则图像时间序列张量将以半精度存储在RAM中，以提高效率。
            folds (list, 可选): 整数列表，指定要加载5个官方分组中的哪一个。若为None，则加载所有分组。
            reference（str，格式：‘YYYY-MM-DD’）：参考日期，根据此数据会生成观测日期序列（以自参考日期以来的天数表示）。
                该日期序列用于基于注意力的方法中的位置编码。
            class_mapping (dict, 可选): 用于定义默认18类别与另一个分类之间映射的字典。
            mono_date (int or str, 可选): 用于指定加载影像时间序列中的一幅。若为int，则为加载日期的位置。
                若为字符串（格式：’YYYY-MM-DD’），则加载指定日期最接近的影像。
            sats (list)：指定要使用的卫星（V1.0中只有Sentinel-2可用）。
        """
        super(PASTIS_Dataset, self).__init__()
        self.folder = folder
        self.norm = norm
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.cache = cache
        self.mem16 = mem16
        self.mono_date = (
            datetime(*map(int, mono_date.split("-")))
            if mono_date and "-" in mono_date
            else None
        )
        self.memory = {}
        self.memory_dates = {}
        self.class_mapping = (
            np.vectorize(lambda x: class_mapping[x])
            if class_mapping is not None
            else class_mapping
        )
        self.target = target
        self.sats = sats

        # 读取PASTIS数据集的元数据
        # GeoJSON是一种地理数据结构编码的格式，基于JSON的地理空间信息数据交换格式。
        print("Reading patch metadata . . .")
        self.meta_patch = gpd.read_file(
            os.path.join(folder, "metadata.geojson"))  # 读取观测地块的元数据信息，关键字段：Flod、ID_PATCH和dates-S2
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)  # 用地块ID做要素索引
        self.meta_patch.sort_index(inplace=True)  # 用新的索引排序

        self.date_tables = {s: None for s in sats}  # 按照卫星类型整理观测日期，字典类型，key：卫星类型；value：日期表
        self.date_range = np.array(range(-200, 600))  # 日期表的列（表示观测日期与参考日期的天数差）
        for s in sats:  # 遍历卫星类型，生成对应的日期表
            dates = self.meta_patch["dates-{}".format(s)]  # 获取每个地块对应的日期序列
            date_table = pd.DataFrame(  # 日期表，index为地块ID，column为天数差（-200~599）
                index=self.meta_patch.index, columns=self.date_range, dtype=int
            )
            for pid, date_seq in dates.items():  # 遍历每个地块的日期序列
                d = pd.DataFrame().from_dict(date_seq, orient="index")  # 将字典转成表格，方便统一计算与参考日期的天数差
                d = d[0].apply(
                    lambda x: (
                            datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
                            - self.reference_date
                    ).days
                )
                date_table.loc[pid, d.values] = 1  # 在日期表中，根据地块ID，将天数差对应的列设为1
            date_table = date_table.fillna(0)  # 将未观测的天数差填充为0
            self.date_tables[s] = {  # date_tables字典（key：卫星名，value：地块观测日期映射表（dict，key：地块ID，value：日期差数组））
                index: np.array(list(d.values()))
                for index, d in date_table.to_dict(orient="index").items()
            }

        print("Done.")

        # 筛选出指定分组中的样本
        if folds is not None:
            self.meta_patch = pd.concat(
                [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
            )

        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index  # 获取指定分组下的所有地块ID

        # 获取规范化的值
        if norm:
            self.norm = {}
            for s in self.sats:
                with open(
                        os.path.join(folder, "NORM_{}_patch.json".format(s)), "r"
                ) as file:
                    normvals = json.loads(file.read())
                selected_folds = folds if folds is not None else range(1, 6)
                means = [normvals["Fold_{}".format(f)]["mean"] for f in selected_folds]  # 获取各个通道的均值
                stds = [normvals["Fold_{}".format(f)]["std"] for f in selected_folds]  # 获取各个通道的标准差
                self.norm[s] = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)  # 计算各个通道，均值和标准差的均值
                self.norm[s] = (  # 转成张量
                    torch.from_numpy(self.norm[s][0]).float(),
                    torch.from_numpy(self.norm[s][1]).float(),
                )
        else:
            self.norm = None
        print("Dataset ready.")

    def __len__(self):
        return self.len

    def get_dates(self, id_patch, sat):
        return self.date_range[np.where(self.date_tables[sat][id_patch] == 1)[0]]

    def __getitem__(self, item):
        id_patch = self.id_patches[item]

        # 检索和准备卫星数据
        if not self.cache or item not in self.memory.keys():
            data = {
                satellite: np.load(
                    os.path.join(
                        self.folder,
                        "DATA_{}".format(satellite),
                        "{}_{}.npy".format(satellite, id_patch),
                    )
                ).astype(np.float32)
                for satellite in self.sats
            }  # 根据地块ID，读取地块npy数据文件，T（时间） x C（通道） x H x W 数组
            data = {s: torch.from_numpy(a) for s, a in data.items()}  # 将数组转成Tensor类型

            if self.norm is not None:  # 对每个通道进行标准化
                data = {
                    s: (d - self.norm[s][0][None, :, None, None])
                       / self.norm[s][1][None, :, None, None]
                    for s, d in data.items()
                }

            if self.target == "semantic":
                target = np.load(
                    os.path.join(
                        self.folder, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch)
                    )
                )  # 读取地块地块的语义分割结果，3（RGB） x H x W 数组
                target = torch.from_numpy(target[0].astype(int))  # 第一个通道的值对应该像素的类别

                if self.class_mapping is not None:
                    target = self.class_mapping(target)

            elif self.target == "instance":
                heatmap = np.load(
                    os.path.join(
                        self.folder,
                        "INSTANCE_ANNOTATIONS",
                        "HEATMAP_{}.npy".format(id_patch),
                    )
                )

                instance_ids = np.load(
                    os.path.join(
                        self.folder,
                        "INSTANCE_ANNOTATIONS",
                        "INSTANCES_{}.npy".format(id_patch),
                    )
                )
                pixel_to_object_mapping = np.load(
                    os.path.join(
                        self.folder,
                        "INSTANCE_ANNOTATIONS",
                        "ZONES_{}.npy".format(id_patch),
                    )
                )

                pixel_semantic_annotation = np.load(
                    os.path.join(
                        self.folder, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch)
                    )
                )

                if self.class_mapping is not None:
                    pixel_semantic_annotation = self.class_mapping(
                        pixel_semantic_annotation[0]
                    )
                else:
                    pixel_semantic_annotation = pixel_semantic_annotation[0]

                size = np.zeros((*instance_ids.shape, 2))
                object_semantic_annotation = np.zeros(instance_ids.shape)
                for instance_id in np.unique(instance_ids):
                    if instance_id != 0:
                        h = (instance_ids == instance_id).any(axis=-1).sum()
                        w = (instance_ids == instance_id).any(axis=-2).sum()
                        size[pixel_to_object_mapping == instance_id] = (h, w)
                        object_semantic_annotation[
                            pixel_to_object_mapping == instance_id
                            ] = pixel_semantic_annotation[instance_ids == instance_id][0]

                target = torch.from_numpy(
                    np.concatenate(
                        [
                            heatmap[:, :, None],  # 0
                            instance_ids[:, :, None],  # 1
                            pixel_to_object_mapping[:, :, None],  # 2
                            size,  # 3-4
                            object_semantic_annotation[:, :, None],  # 5
                            pixel_semantic_annotation[:, :, None],  # 6
                        ],
                        axis=-1,
                    )
                ).float()

            if self.cache:
                if self.mem16:
                    self.memory[item] = [{k: v.half() for k, v in data.items()}, target]
                else:
                    self.memory[item] = [data, target]

        else:
            data, target = self.memory[item]
            if self.mem16:
                data = {k: v.float() for k, v in data.items()}

        # 检索地块观测日期的序列（与参考日期的天数差）
        if not self.cache or id_patch not in self.memory_dates.keys():
            dates = {
                s: torch.from_numpy(self.get_dates(id_patch, s)) for s in self.sats
            }
            if self.cache:
                self.memory_dates[id_patch] = dates
        else:
            dates = self.memory_dates[id_patch]

        if self.mono_date is not None:  #
            if isinstance(self.mono_date, int):
                data = {s: data[s][self.mono_date].unsqueeze(0) for s in self.sats}
                dates = {s: dates[s][self.mono_date] for s in self.sats}
            else:
                mono_delta = (self.mono_date - self.reference_date).days
                mono_date = {
                    s: int((dates[s] - mono_delta).abs().argmin()) for s in self.sats
                }
                data = {s: data[s][mono_date[s]].unsqueeze(0) for s in self.sats}
                dates = {s: dates[s][mono_date[s]] for s in self.sats}

        if self.mem16:
            data = {k: v.float() for k, v in data.items()}

        if len(self.sats) == 1:
            data = data[self.sats[0]]
            dates = dates[self.sats[0]]

        return (data, dates), target


def prepare_dates(date_dict, reference_date):
    d = pd.DataFrame().from_dict(date_dict, orient="index")
    d = d[0].apply(
        lambda x: (
                datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
                - reference_date
        ).days
    )
    return d.values


def compute_norm_vals(folder, sat):
    norm_vals = {}
    for fold in range(1, 6):
        dt = PASTIS_Dataset(folder=folder, norm=False, folds=[fold], sats=[sat])
        means = []
        stds = []
        for i, b in enumerate(dt):
            print("{}/{}".format(i, len(dt)), end="\r")
            data = b[0][0][sat]  # T x C x H x W
            data = data.permute(1, 0, 2, 3).contiguous()  # C x B x T x H x W
            means.append(data.view(data.shape[0], -1).mean(dim=-1).numpy())
            stds.append(data.view(data.shape[0], -1).std(dim=-1).numpy())

        mean = np.stack(means).mean(axis=0).astype(float)
        std = np.stack(stds).mean(axis=0).astype(float)

        norm_vals["Fold_{}".format(fold)] = dict(mean=list(mean), std=list(std))

    with open(os.path.join(folder, "NORM_{}_patch.json".format(sat)), "w") as file:
        file.write(json.dumps(norm_vals, indent=4))
