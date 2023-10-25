import argparse
import json
import os
import pickle as pkl
import pprint
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchnet as tnt
from PIL import Image
from src import utils
from src import model_utils
from src.dataset import SITS_Dataset
from src.learning.metrics import confusion_matrix_analysis
from src.learning.miou import IoU
from src.learning.weight_init import weight_init

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default='stenn',
    type=str,
    help="指定网络：utae, unet3d, convlstm, convgru, buconvlstm, uconvlstm, vgg, stenn, stenn_nodense, stenn_notransformer",
)
parser.add_argument(
    "--dataset_folder",
    default="/root/autodl-tmp/PASTIS",
    type=str,
    help="卫星影像时间序列数据集的存放路径"
)
parser.add_argument(
    "--result_folder",
    default="./train_result",
    type=str,
    help="训练结果的保存路径"
)
parser.add_argument(
    "--rdm_seed",
    default=1,
    type=int,
    help="随机种子"
)
parser.add_argument(
    "--device",
    default="cuda",
    type=str,
    help="指定训练平台：cuda/cpu"
)
parser.add_argument(
    "--display_step",
    default=50,
    type=int,
    help="打印训练指标的批次间隔",
)
parser.add_argument(
    "--val_every",
    default=5,
    type=int,
    help="验证的迭代间隔",
)
parser.add_argument(
    "--val_after",
    default=0,
    type=int,
    help="指定迭代多少次后才进行验证",
)
parser.add_argument("--epochs", default=100, type=int, help="迭代次数")
parser.add_argument("--batch_size", default=1, type=int, help="批次大小")
parser.add_argument("--lr", default=0.001, type=float, help="学习率")
parser.add_argument("--num_classes", default=20, type=int, help="类别数量")
parser.add_argument("--ignore_index", default=-1, type=int, help="忽略类别")
parser.add_argument("--input_channel", default=10, type=int, help="影像通道数")
parser.add_argument("--fold", default=1, type=int, help="指定训练、验证和测试的分组（1~5）")
parser.add_argument("--pad_value", default=0, type=float, help="时间序列长度不一时，padding的值")


# 通过递归将数据加载到指定设备上
def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(v, device) for v in x]


def save_test_results(out, patch_ID, config):
    result = torch.softmax(out, dim=1)
    result = torch.argmax(result, 1)
    for i in range(result.shape[0]):
        obj = result[i].to('cpu', torch.uint8).numpy()
        im = Image.fromarray(obj)
        im.save(os.path.join(config.res_dir, config.model, "Test", "{}.png".format(patch_ID[i])))


def iterate(model, data_loader, criterion, config, device=None, mode="train", optimizer=None, test=False):
    loss_meter = tnt.meter.AverageValueMeter()
    iou_meter = IoU(  # 语义分割评估指标，计算每个类的IoU和平均IoU
        num_classes=config.num_classes,
        ignore_index=config.ignore_index,
        cm_device=config.device,
    )
    start_time = time.time()
    for i, batch in enumerate(data_loader):
        if device is not None:
            batch = recursive_todevice(batch, device)
        (data, dates), target, patch_ID = batch
        target = target.long()

        if mode == "train":
            optimizer.zero_grad()  # 清空梯度
            out = model(data, batch_positions=dates)
        else:
            with torch.no_grad():  # 将模型的所有参数的requires_grad设置为False
                out = model(data, batch_positions=dates)
                if test: save_test_results(out, patch_ID, config)

        loss = criterion(out, target)
        if mode == "train":
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新网络参数

        with torch.no_grad():
            pred = out.argmax(dim=1)
        iou_meter.add(pred, target)
        loss_meter.add(loss.item())

        if (i + 1) % config.display_step == 0:
            miou, acc = iou_meter.get_miou_acc()
            print(
                "Step [{}/{}], Loss: {:.4f}, Acc: {:.2f}, mIoU: {:.2f}".format(
                    i + 1, len(data_loader), loss_meter.value()[0], acc, miou
                )
            )
    end_time = time.time()
    total_time = end_time - start_time
    print("Epoch time:{:.1f}".format(total_time))
    miou, acc = iou_meter.get_miou_acc()
    metrics = {
        "{}_accuracy".format(mode): acc,
        "{}_loss".format(mode): loss_meter.value()[0],
        "{}_IoU".format(mode): miou,
        "{}_epoch_time".format(mode): total_time,
    }
    if mode == "test":
        return metrics, iou_meter.conf_metric.value()  # confusion matrix
    else:
        return metrics


# 创建保存结果的文件夹
def prepare_output(config):
    os.makedirs(config.result_folder, exist_ok=True)
    for fold in range(1, 6):
        os.makedirs(os.path.join(config.result_folder, "Fold_{}".format(fold)), exist_ok=True)


# 记录训练日志
def checkpoint(fold, log, config):
    with open(
            os.path.join(config.result_folder, "Fold_{}".format(fold), "trainlog.json"), "w"
    ) as outfile:
        json.dump(log, outfile, indent=4)


# 保存训练参数
def save_results(fold, metrics, conf_mat, config):
    with open(
            os.path.join(config.result_folder, "Fold_{}".format(fold), "test_metrics.json"), "w"
    ) as outfile:
        json.dump(metrics, outfile, indent=4)
    pkl.dump(
        conf_mat,
        open(
            os.path.join(config.result_folder, "Fold_{}".format(fold), "conf_mat.pkl"), "wb"
        ),
    )


# 整体性能评估
def overall_performance(config):
    cm = np.zeros((config.num_classes, config.num_classes))
    for fold in range(1, 6):
        cm += pkl.load(
            open(
                os.path.join(config.result_folder, "Fold_{}".format(fold), "conf_mat.pkl"),
                "rb",
            )
        )

    if config.ignore_index is not None:
        cm = np.delete(cm, config.ignore_index, axis=0)
        cm = np.delete(cm, config.ignore_index, axis=1)

    _, perf = confusion_matrix_analysis(cm)

    print("Overall performance:")
    print("Acc: {},  IoU: {}".format(perf["Accuracy"], perf["MACRO_IoU"]))

    with open(os.path.join(config.result_folder, "overall.json"), "w") as file:
        file.write(json.dumps(perf, indent=4))


def main(config):
    fold_sequence = [
        [[1, 2, 3], [4], [5]],
        [[2, 3, 4], [5], [1]],
        [[3, 4, 5], [1], [2]],
        [[4, 5, 1], [2], [3]],
        [[5, 1, 2], [3], [4]],
    ]
    prepare_output(config)
    np.random.seed(config.rdm_seed)
    torch.manual_seed(config.rdm_seed)
    device = torch.device(config.device)
    fold_sequence = (  # 若未指定分组，则进行五组训练
        fold_sequence if config.fold is None else [fold_sequence[config.fold - 1]]
    )
    for fold, (train_folds, val_folds, test_folds) in enumerate(fold_sequence):
        if config.fold is not None:
            fold = config.fold - 1
        train_dataset = SITS_Dataset(config.dataset_folder, folds=train_folds)
        val_dataset = SITS_Dataset(config.dataset_folder, folds=val_folds)
        test_dataset = SITS_Dataset(config.dataset_folder, folds=test_folds)
        collate_fn = lambda x: utils.pad_collate(x, pad_value=config.pad_value)  # 重写对Batch数据的堆叠方式
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        val_loader = data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        print("Train {}, Val {}, Test {}".format(len(train_dataset), len(val_dataset), len(test_dataset)))

        model = model_utils.get_model(config)
        print(model)
        config.N_params = utils.get_ntrainparams(model)  # 获取模型中可训练的参数
        print("TOTAL TRAINABLE PARAMETERS :", config.N_params)
        with open(os.path.join(config.result_folder, "conf.json"), "w") as file:  # 将参数写到JSON文件中
            file.write(json.dumps(vars(config), indent=4))
        model = model.to(device)
        model_file = os.path.join(config.result_folder, "Fold_{}".format(fold + 1), "model.pth.tar")
        if os.path.exists(model_file):  # 初始化模型参数
            model.load_state_dict(
                torch.load(model_file, map_location=device)["state_dict"]
            )
        else:
            model.apply(weight_init)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        weight = torch.ones(config.num_classes, device=device).float()
        weight[config.ignore_index] = 0
        criterion = nn.CrossEntropyLoss(weight=weight)
        trainlog = {}
        best_mIoU = 0  # 最佳交并比
        for epoch in range(1, config.epochs + 1):
            print("EPOCH {}/{}".format(epoch, config.epochs))
            model.train()
            train_metrics = iterate(
                model,
                data_loader=train_loader,
                criterion=criterion,
                config=config,
                device=device,
                optimizer=optimizer,
            )
            if epoch % config.val_every == 0 and epoch > config.val_after:
                print("Validation . . . ")
                model.eval()
                val_metrics = iterate(
                    model,
                    data_loader=val_loader,
                    criterion=criterion,
                    config=config,
                    device=device,
                    optimizer=optimizer,
                    mode="val",
                )
                print(
                    "Loss {:.4f},  Acc {:.2f},  IoU {:.4f}".format(
                        val_metrics["val_loss"],
                        val_metrics["val_accuracy"],
                        val_metrics["val_IoU"],
                    )
                )
                trainlog[epoch] = {**train_metrics, **val_metrics}
                checkpoint(fold + 1, trainlog, config)
                if val_metrics["val_IoU"] >= best_mIoU:
                    best_mIoU = val_metrics["val_IoU"]
                    torch.save(
                        {
                            "epoch": epoch,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        model_file,
                    )
            else:
                trainlog[epoch] = {**train_metrics}
                checkpoint(fold + 1, trainlog, config)
        print("Testing best epoch . . .")
        model.load_state_dict(
            torch.load(model_file)["state_dict"]
        )
        model.eval()

        test_metrics, conf_mat = iterate(
            model,
            data_loader=test_loader,
            criterion=criterion,
            config=config,
            device=device,
            optimizer=optimizer,
            mode="test",
        )
        print(
            "Loss {:.4f},  Acc {:.2f},  IoU {:.4f}".format(
                test_metrics["test_loss"],
                test_metrics["test_accuracy"],
                test_metrics["test_IoU"],
            )
        )
        save_results(fold + 1, test_metrics, conf_mat.cpu().numpy(), config)


if __name__ == "__main__":
    config = parser.parse_args()
    print(config)
    main(config)
    os.system("/usr/bin/shutdown")
