#!/usr/bin/env python
# -*- coding:utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTripleLoss(nn.Module):

    """Qi Qian, et al.,
    `SoftTriple Loss: Deep Metric Learning Without Triplet Sampling`,
    https://arxiv.org/abs/1909.05235
    """

    def __init__(
        self,
        embedding_dim: int,
        num_categories: int,
        num_initial_center: int = 2,
        similarity_margin: float = 0.1,
        coef_regularizer1: float = 1e-2,
        coef_regularizer2: float = 1e-2,
        coef_scale_softmax: float = 1.0,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        """Constructor

        Args:
            embedding_dim: dimension of inputs to this module (N x embedding_dim)
            num_categories: total category count to classify
            num_initial_center: initial number of centers for each categories
            similarity_margin: margin term as is in triplet loss
            coef_regularizer1: entropy regularizer for dictibution over classes
            coef_regularizer2: regularizer for cluster variancce.
            coef_scale_softmax: scaling factor before final softmax op
            device: device on which this loss is computed
        """

        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_categories = num_categories
        self.num_initial_center = num_initial_center
        self.delta = similarity_margin
        self.gamma_inv = 1 / coef_regularizer1
        self.tau = coef_regularizer2
        self.lambda_ = coef_scale_softmax
        self.device = device
        self.fc_hidden = nn.Linear(
            embedding_dim, num_categories * num_initial_center
        ).to(device)
        nn.init.xavier_normal_(self.fc_hidden.weight)
        self.base_loss = nn.CrossEntropyLoss().to(device)
        self.softmax = nn.Softmax(dim=2).to(device)

    def infer(self, embedding):
        weight = F.normalize(self.fc_hidden.weight)
        x = F.linear(embedding, weight).view(
            -1, self.num_categories, self.num_initial_center
        )
        x = self.softmax(x.mul(self.gamma_inv)).mul(x).sum(dim=2)
        return x

    def cluster_variance_loss(self):
        weight = F.normalize(self.fc_hidden.weight)
        loss = 0.0
        for i in range(self.num_categories):
            weight_sub = weight[
                i * self.num_initial_center : (i + 1) * self.num_initial_center
            ]
            subtraction_norm = 1.0 - torch.matmul(
                weight_sub, weight_sub.transpose(1, 0)
            )
            subtraction_norm[subtraction_norm < 0.0] = 0
            loss += torch.sqrt(2 * subtraction_norm.triu(diagonal=1)).sum()

        loss /= (
            self.num_categories
            * self.num_initial_center
            * (self.num_initial_center - 1)
        )
        return loss

    def forward(self, embeddings, labels):
        h = self.infer(embeddings)
        one_hot = torch.zeros(h.size(), device=self.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        h = h - self.delta * one_hot
        h.mul_(self.lambda_)
        clf_loss = self.base_loss(h, labels)
        var_loss = self.cluster_variance_loss()
        return clf_loss + self.tau * var_loss
