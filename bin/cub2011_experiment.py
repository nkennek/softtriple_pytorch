#!/usr/bin/env python
# -*- coding:utf-8 -*-


import argparse
import subprocess
import shutil
import os

import pandas as pd
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import inception_v3
from torchvision.models import googlenet
from tqdm import tqdm

from softtriple_pytorch import SoftTripleLoss


class CUB2011(ImageFolder):
    def __init__(self, image_subset, **kwargs):
        """Constructor
        Args:
            image_subset: `train` or `test`
            kwargs: arguments in super class
        """

        dataset_parent_path = f"{os.environ['HOME']}/.local/"
        if not os.path.exists(dataset_parent_path):
            os.mkdir(dataset_parent_path)

        if "CUB_200_2011" not in os.listdir(dataset_parent_path):
            subprocess.check_output(
                [
                    f"{os.path.dirname(os.path.abspath(__file__))}/download_cub2011.sh",
                    f"{dataset_parent_path}",
                ],
                shell=True,
            )

        images = pd.read_csv(
            os.path.join(dataset_parent_path, "CUB_200_2011", "images.txt"),
            delimiter=" ",
            index_col=0,
        ).iloc[:, 0]
        image_class_labels = pd.read_csv(
            os.path.join(dataset_parent_path, "CUB_200_2011", "image_class_labels.txt"),
            delimiter=" ",
            index_col=0,
        ).iloc[:, 0]
        if image_subset == "train":
            images = images[image_class_labels <= 100]
        elif image_subset == "test":
            images = images[image_class_labels > 100]
        else:
            raise ValueError("Unknown subset. It should be either train or test")

        # create directory for dataset by symlink
        tmp_directory = os.path.join(dataset_parent_path, "CUB_200_2011", "images_tmp")
        if os.path.exists(tmp_directory):
            shutil.rmtree(tmp_directory)

        os.mkdir(tmp_directory)
        for image_path in images.values:
            dirname = os.path.basename(os.path.dirname(image_path))
            dirname_abs = os.path.join(tmp_directory, dirname)
            if not os.path.exists(dirname_abs):
                os.mkdir(dirname_abs)

            os.symlink(
                os.path.join(dataset_parent_path, "CUB_200_2011", "images", image_path),
                os.path.join(dirname_abs, os.path.basename(image_path)),
            )

        super().__init__(root=tmp_directory, **kwargs)


class BackBoneEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        classifier = googlenet(pretrained=True)
        self.transform_input = classifier.transform_input
        for layer_name in (
            "conv1",
            "maxpool1",
            "conv2",
            "conv3",
            "maxpool2",
            "inception3a",
            "inception3b",
            "maxpool3",
            "inception4a",
            "inception4b",
            "inception4c",
            "inception4d",
            "inception4e",
            "maxpool4",
            "inception5a",
            "inception5b",
            "avgpool",
        ):
            layer = getattr(classifier, layer_name)
            setattr(self, layer_name, layer)

        self.fc = nn.Linear(1024, embedding_dim)
        self.relu = nn.ReLU(inplace=True)

    def freeze_bn_grad(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def unfreeze_bn_grad(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = True
                m.bias.requires_grad = True

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)

        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.fc(x)
        x = self.relu(x)
        # x = self.bn_last(x)
        x = F.normalize(x)
        return x


transform_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ]
)

transform_test = transforms.Compose(
    [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()]
)


def _iterate_train(batch, backbone, center, optimizer1, optimzier2, device):
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    backbone.train()
    center.train()

    x, labels = batch
    x = x.to(device)
    labels = labels.to(device)
    feat = backbone(x)
    with torch.no_grad():
        pred = center.infer(feat)
        pred_indices = torch.argmax(pred, dim=1)
        correct = torch.eq(pred_indices, labels).view(-1)

    loss = center(feat, labels)
    loss.backward()
    optimizer1.step()
    optimizer2.step()

    return correct, loss.item()


def _iterate_test(batch, backbone, center, device):
    with torch.no_grad():
        backbone.eval()
        center.eval()
        x, labels = batch
        x = x.to(device)
        labels = labels.to(device)
        feat = backbone(x)
        pred = center.infer(feat)
        pred_indices = torch.argmax(pred, dim=1)
        correct = torch.eq(pred_indices, labels).view(-1)
        loss = center(feat, labels)

        return correct, loss.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CUB2011 training")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--lr_center", type=float, default=1e-2)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--max_epoch", "-e", type=int, default=50)
    parser.add_argument("--validation_size", "-v", type=float, default=0.01)
    parser.add_argument("--embedding_dim", "-emb", type=int, default=64)
    parser.add_argument(
        "--k", type=int, default=10, help="number of centers for each categories"
    )
    parser.add_argument(
        "--margin", type=float, default=0.01, help="margin term in triplet loss"
    )
    parser.add_argument("--gamma", type=float, default=0.1, help="gamma in the paper")
    parser.add_argument("--tau", type=float, default=0.2, help="tau in the paper")
    parser.add_argument("--lamb", type=float, default=5.0, help="lambda in the paper")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dataset = CUB2011("train", transform=transform_train)
    val_size = int(len(dataset) * args.validation_size)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [len(dataset) - val_size, val_size]
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    backbone = BackBoneEmbedding(args.embedding_dim).to(device)
    softtriple = SoftTripleLoss(
        args.embedding_dim,
        len(dataset.classes),
        args.k,
        args.margin,
        args.gamma,
        args.tau,
        args.lamb,
        device,
    )
    backbone.freeze_bn_grad()
    optimizer1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, backbone.parameters()), lr=args.lr_backbone
    )
    optimizer2 = torch.optim.Adam(softtriple.parameters(), lr=args.lr_center)
    backbone.unfreeze_bn_grad()
    lr_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=20, gamma=0.1)
    lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=20, gamma=0.1)

    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    # mlflow
    with mlflow.start_run() as run:
        # Log args into mlflow
        for key, value in vars(args).items():
            mlflow.log_param(key, value)

        for epoch in range(args.max_epoch):
            print(f"epoch {epoch+1}")
            loss = 0
            correct = 0
            for idx, batch in tqdm(enumerate(train_dl)):
                correct_iter, loss_iter = _iterate_train(
                    batch, backbone, softtriple, optimizer1, optimizer2, device
                )
                correct += correct_iter.sum()
                loss = (idx * loss + loss_iter) / (idx + 1)

            accuracy = float(correct) / len(train_dataset)
            print(f"\t Train Loss: {loss}")
            print(f"\t Train Accuracy: {accuracy}")
            mlflow.log_metric("train-loss", loss, step=epoch + 1)

            correct = 0
            loss = 0
            for idx, batch in enumerate(val_dl):
                correct_iter, loss_iter = _iterate_test(
                    batch, backbone, softtriple, device
                )
                correct += correct_iter.sum()
                loss = (idx * loss + loss_iter) / (idx + 1)

            accuracy = float(correct) / len(val_dataset)
            print(f"Validation:\nLoss:{loss}\nAccuracy:{accuracy}")
            mlflow.log_metric("val-loss", loss, step=epoch + 1)
            mlflow.log_metric("val-acc", accuracy, step=epoch + 1)

            lr_scheduler1.step()
            lr_scheduler2.step()
            print(f"-----------------------------------")

        # Log model
        mlflow.pytorch.log_model(backbone, "backbone")
        mlflow.pytorch.log_model(softtriple, "center")
