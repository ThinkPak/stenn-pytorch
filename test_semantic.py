import argparse
import json
import os
import pprint
import pickle as pkl

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from src import utils, model_utils
from src.dataset import SITS_Dataset

from train_semantic import iterate, overall_performance, prepare_output

parser = argparse.ArgumentParser()
# Model parameters
parser.add_argument(
    "--weight_folder",
    default="./train_result/Model Train/stenn",
    type=str,
    help="模型权重所在文件夹",
)
parser.add_argument(
    "--dataset_folder",
    default="/root/autodl-tmp/PASTIS",
    type=str,
    help="卫星影像时间序列数据集的存放路径",
)
parser.add_argument(
    "--res_dir",
    default="./train_result/Model Inference",
    type=str,
    help="预测结果保存路径"
)
parser.add_argument(
    "--num_workers",
    default=8,
    type=int,
    help="Number of data loading workers",
)
parser.add_argument(
    "--fold",
    default=1,
    type=int,
    help="指定训练、验证和测试的分组（1~5）",
)
parser.add_argument(
    "--device",
    default="cuda",
    type=str,
    help="指定训练平台：cuda/cpu",
)
parser.add_argument(
    "--display_step",
    default=50,
    type=int,
    help="打印训练指标的批次间隔",
)


def save_results(fold, metrics, conf_mat, config):
    with open(
            os.path.join(config.res_dir, config.model, "Fold_{}".format(fold), "test_metrics.json"), "w"
    ) as outfile:
        json.dump(metrics, outfile, indent=4)
    pkl.dump(
        conf_mat,
        open(
            os.path.join(config.res_dir, config.model, "Fold_{}".format(fold), "conf_mat.pkl"), "wb"
        ),
    )


def main(config):
    fold_sequence = [
        [[1, 2, 3], [4], [5]],
        [[2, 3, 4], [5], [1]],
        [[3, 4, 5], [1], [2]],
        [[4, 5, 1], [2], [3]],
        [[5, 1, 2], [3], [4]],
    ]

    np.random.seed(config.rdm_seed)
    torch.manual_seed(config.rdm_seed)
    device = torch.device(config.device)
    prepare_output(config)
    os.makedirs(os.path.join(config.res_dir, config.model, "Fold_{}".format(config.fold)), exist_ok=True)
    os.makedirs(os.path.join(config.res_dir, config.model, "Test"), exist_ok=True)

    model = model_utils.get_model(config, mode="semantic")
    model = model.to(device)

    config.N_params = utils.get_ntrainparams(model)
    print(model)
    print("TOTAL TRAINABLE PARAMETERS :", config.N_params)

    fold_sequence = (
        fold_sequence if config.fold is None else [fold_sequence[config.fold - 1]]
    )
    for fold, (train_folds, val_fold, test_fold) in enumerate(fold_sequence):
        if config.fold is not None:
            fold = config.fold - 1

        # Dataset definition
        dt_test = SITS_Dataset(folder=config.dataset_folder, folds=test_fold)
        collate_fn = lambda x: utils.pad_collate(x, pad_value=config.pad_value)
        test_loader = data.DataLoader(
            dt_test,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        # Load weights
        sd = torch.load(
            os.path.join(config.weight_folder, "Fold_{}".format(fold + 1), "model.pth.tar"),
            map_location=device,
        )
        model.load_state_dict(sd["state_dict"])

        # Loss
        weights = torch.ones(config.num_classes, device=device).float()
        weights[config.ignore_index] = 0
        criterion = nn.CrossEntropyLoss(weight=weights)

        # Inference
        print("Testing . . .")
        model.eval()
        test_metrics, conf_mat = iterate(
            model,
            data_loader=test_loader,
            criterion=criterion,
            config=config,
            optimizer=None,
            mode="test",
            device=device,
            test=True,
        )
        print(
            "Loss {:.4f},  Acc {:.2f},  IoU {:.4f}".format(
                test_metrics["test_loss"],
                test_metrics["test_accuracy"],
                test_metrics["test_IoU"],
            )
        )
        save_results(fold + 1, test_metrics, conf_mat.cpu().numpy(), config)

    if config.fold is None:
        overall_performance(config)


if __name__ == "__main__":
    test_config = parser.parse_args()

    with open(os.path.join(test_config.weight_folder, "conf.json")) as file:
        model_config = json.loads(file.read())

    config = {**model_config, **vars(test_config)}
    config = argparse.Namespace(**config)
    config.fold = test_config.fold

    pprint.pprint(config)
    main(config)
