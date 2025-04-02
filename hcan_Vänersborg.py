#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tsai.all import *
import sklearn.metrics as skm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------- 位置编码模块 -----------------
class tAPE(nn.Module):
    "time Absolute Position Encoding"

    def __init__(self, d_model: int, seq_len=1024, dropout: float = 0.1, scale_factor=1.0):
        super().__init__()
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # 有助于模型更好地理解序列中元素的位置关系
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 让模型能够理解序列中不同位置的元素之间的相对位置关系
        pe[:, 0::2] = torch.sin((position * div_term) * (d_model / seq_len))
        pe[:, 1::2] = torch.cos((position * div_term) * (d_model / seq_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):  # x: [batch, seq_len, d_model]
        return self.dropout(x + self.pe)


class AbsolutePositionalEncoding(nn.Module):
    "Absolute positional encoding"

    def __init__(self, d_model: int, seq_len=1024, dropout: float = 0.1, scale_factor=1.0):
        super().__init__()
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(x + self.pe)


class LearnablePositionalEncoding(nn.Module):
    "Learnable positional encoding"

    def __init__(self, d_model: int, seq_len=1024, dropout: float = 0.1):
        super().__init__()
        self.pe = nn.Parameter(torch.empty(seq_len, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(x + self.pe)


# ----------------- 多头自注意力模块 -----------------
class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.01):
        super().__init__()
        self.n_heads = n_heads
        self.scale = d_model ** -0.5
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(d_model)

    def forward(self, x):  # x: [batch, seq_len, d_model]
        bs, seq_len, _ = x.shape
        k = self.key(x).reshape(bs, seq_len, self.n_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(bs, seq_len, self.n_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(bs, seq_len, self.n_heads, -1).transpose(1, 2)
        attn = F.softmax(torch.matmul(q, k) * self.scale, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(bs, seq_len, -1)
        return self.to_out(out)

# 通过引入相对位置偏置来增强模型对序列中元素相对位置的理解能力
class Attention_Rel_Scl(nn.Module):
    def __init__(self, d_model: int, seq_len: int, n_heads: int = 8, dropout: float = 0.01):
        super().__init__()
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.scale = d_model ** -0.5
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.relative_bias_table = nn.Parameter(torch.zeros((2 * self.seq_len - 1), n_heads))
        # 生成一个行向量，该行向量的每个元素代表序列中某个位置的索引
        coords = torch.meshgrid((torch.arange(1), torch.arange(self.seq_len)), indexing="xy")
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[1] += self.seq_len - 1
        relative_coords = relative_coords.permute(1, 2, 0)
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(d_model)

    def forward(self, x):
        bs, seq_len, _ = x.shape
        k = self.key(x).reshape(bs, seq_len, self.n_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(bs, seq_len, self.n_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(bs, seq_len, self.n_heads, -1).transpose(1, 2)
        attn = F.softmax(torch.matmul(q, k) * self.scale, dim=-1)
        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, self.n_heads))
        relative_bias = relative_bias.reshape(self.seq_len, self.seq_len, -1).permute(2, 0, 1).unsqueeze(0)
        attn = attn + relative_bias
        out = torch.matmul(attn, v).transpose(1, 2).reshape(bs, seq_len, -1)
        return self.to_out(out)


class Attention_Rel_Vec(nn.Module):
    def __init__(self, d_model: int, seq_len: int, n_heads: int = 8, dropout: float = 0.01):
        super().__init__()
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.scale = d_model ** -0.5
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.Er = nn.Parameter(torch.randn(self.seq_len, int(d_model / n_heads)))
        self.register_buffer("mask", torch.tril(torch.ones(self.seq_len, self.seq_len)).unsqueeze(0).unsqueeze(0))
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(d_model)

    def forward(self, x):
        bs, seq_len, _ = x.shape
        k = self.key(x).reshape(bs, seq_len, self.n_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(bs, seq_len, self.n_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(bs, seq_len, self.n_heads, -1).transpose(1, 2)
        QEr = torch.matmul(q, self.Er.transpose(0, 1))
        Srel = self.skew(QEr)
        attn = F.softmax((torch.matmul(q, k) + Srel) * self.scale, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(bs, seq_len, -1)
        return self.to_out(out)

    def skew(self, QEr):
        padded = F.pad(QEr, (1, 0))
        bs, n_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(bs, n_heads, num_cols, num_rows)
        return reshaped[:, :, 1:, :]


# ----------------- 定义网络骨干 -----------------
class ConvTranBackbone(nn.Module):
    def __init__(self, c_in: int, seq_len: int, d_model=16, n_heads: int = 8, dim_ff: int = 256,
                 abs_pos_encode: str = 'tAPE', rel_pos_encode: str = 'eRPE', dropout: float = 0.01):
        super().__init__()
        self.embed_layer = nn.Sequential(
            nn.Conv2d(1, d_model * 4, kernel_size=[1, 7], padding='same'),
            nn.BatchNorm2d(d_model * 4),
            nn.GELU()
        )
        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(d_model * 4, d_model, kernel_size=[c_in, 1], padding='valid'),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        assert abs_pos_encode in ['tAPE', 'sin', 'learned', None]
        if abs_pos_encode == 'tAPE':
            self.abs_position = tAPE(d_model, dropout=dropout, seq_len=seq_len)
        elif abs_pos_encode == 'sin':
            self.abs_position = AbsolutePositionalEncoding(d_model, dropout=dropout, seq_len=seq_len)
        elif abs_pos_encode == 'learned':
            self.abs_position = LearnablePositionalEncoding(d_model, dropout=dropout, seq_len=seq_len)
        else:
            self.abs_position = nn.Identity()
        assert rel_pos_encode in ['eRPE', 'vector', None]
        if rel_pos_encode == 'eRPE':
            self.attention_layer = Attention_Rel_Scl(d_model, seq_len, n_heads=n_heads, dropout=dropout)
        elif rel_pos_encode == 'vector':
            self.attention_layer = Attention_Rel_Vec(d_model, seq_len, n_heads=n_heads, dropout=dropout)
        else:
            self.attention_layer = Attention(d_model, n_heads=n_heads, dropout=dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.FeedForward = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):  # x: [batch, c_in, seq_len]
        x = x.unsqueeze(1)  # [batch, 1, c_in, seq_len]
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(2)  # [batch, d_model, seq_len]
        x_src = x_src.permute(0, 2, 1)  # [batch, seq_len, d_model]
        x_src_pos = self.abs_position(x_src)
        att = self.LayerNorm(x_src + self.attention_layer(x_src_pos))
        out = self.LayerNorm2(att + self.FeedForward(att))
        return out.permute(0, 2, 1)  # [batch, d_model, seq_len]


# ----------------- 定义输出头模块 -----------------
class lin_nd_head(nn.Sequential):
    "Module to create a nd output head with linear layers"

    def __init__(self, n_in, n_out, seq_len=None, d=None, flatten=False, use_bn=False, fc_dropout=0.):
        if seq_len is None:
            seq_len = 1
        if d is None:
            fd = 1
            shape = [n_out]
        elif is_listy(d):
            fd = 1
            shape = []
            for _d in d:
                fd *= _d
                shape.append(_d)
            if n_out > 1: shape.append(n_out)
        else:
            fd = d
            shape = [d, n_out] if n_out > 1 else [d]
        layers = []
        if use_bn:
            layers += [nn.BatchNorm1d(n_in)]
        if fc_dropout:
            layers += [nn.Dropout(fc_dropout)]
        if d is None:
            if not flatten or seq_len == 1:
                layers += [nn.AdaptiveAvgPool1d(1), Squeeze(-1), nn.Linear(n_in, n_out)]
                if n_out == 1:
                    layers += [Squeeze(-1)]
            else:
                layers += [Reshape(), nn.Linear(n_in * seq_len, n_out * fd)]
                if n_out * fd == 1:
                    layers += [Squeeze(-1)]
        else:
            if seq_len == 1:
                layers += [nn.AdaptiveAvgPool1d(1)]
            if not flatten and fd == seq_len:
                layers += [Transpose(1, 2), nn.Linear(n_in, n_out)]
            else:
                layers += [Reshape(), nn.Linear(n_in * seq_len, n_out * fd)]
            layers += [Reshape(*shape)]
        super().__init__(*layers)


create_lin_nd_head = lin_nd_head
lin_3d_head = lin_nd_head  # included for backwards compatiblity
create_lin_3d_head = lin_nd_head  # included for backwards compatiblity


# ----------------- 定义混合模型 -----------------
class HybridHCAN(nn.Module):
    def __init__(self, c_in: int, c_out: int, seq_len: int = 500, d_model: int = 16, fc_channels: int = 128,
                 custom_head=None, **kwargs):
        """
        c_in: 输入通道数
        c_out: 分类类别数
        seq_len: 序列长度，默认值设为500（可根据实际数据调整）
        d_model: 时间分支特征维度（ConvTranBackbone 输出的通道数）
        fc_channels: 变量依赖分支（FCNPlus backbone）输出的通道数
        custom_head: 接收 TSClassifier 传入的额外参数（此处不使用）
        """
        super().__init__()
        # 时间依赖分支（自定义，参考论文提取时序信息）
        self.time_branch = ConvTranBackbone(c_in, seq_len, d_model=d_model, n_heads=8, dim_ff=256,
                                            abs_pos_encode='tAPE', rel_pos_encode='eRPE', dropout=0.01)
        # 变量间依赖分支（利用 tsai 的 FCNPlus，这里我们仅使用其 backbone 部分）
        self.var_branch = FCNPlus(c_in, fc_channels)
        # 定义全局池化：对每个分支的输出在时间维度上进行自适应平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # 最终拼接后全连接分类层（输入维度为 d_model + fc_channels）
        self.fc = nn.Linear(d_model + fc_channels, c_out)

    def forward(self, x):
        # 输入 x: [batch, c_in, seq_len]
        # 时间依赖分支
        time_feat = self.time_branch(x)  # 输出 shape: [batch, d_model, seq_len]
        time_feat = self.global_pool(time_feat).squeeze(-1)  # 变为 [batch, d_model]
        # 变量间依赖分支：使用 FCNPlus 的 backbone 部分提取特征
        var_feat = self.var_branch.backbone(x)  # 输出 shape: [batch, fc_channels, seq_len]
        var_feat = self.global_pool(var_feat).squeeze(-1)  # 变为 [batch, fc_channels]
        # 拼接两个分支的特征
        combined = torch.cat([time_feat, var_feat], dim=1)  # shape: [batch, d_model + fc_channels]
        # 经过全连接层输出分类结果
        out = self.fc(combined)
        return out

# ----------------- 主程序入口 -----------------
def main():
    # 加载数据
    df1 = pd.read_csv(r".\dataset\damage_detection_Vänersborg\type00.csv", index_col=0)
    df2 = pd.read_csv(r".\dataset\damage_detection_Vänersborg\type01.csv", index_col=0)
    df3 = pd.read_csv(r".\dataset\damage_detection_Vänersborg\type02.csv", index_col=0)
    X, y = df2xy(pd.concat([df1[:176000], df2[:176000], df3[:176000]], ignore_index=True), data_cols=None)
    X = X.reshape(1056, 500, 3)
    X = np.transpose(X, (0, 2, 1))
    label = [0] * 352 + [1] * 352 + [2] * 352
    y = pd.DataFrame(label)
    splits = get_splits(y, n_splits=1, valid_size=0.25, test_size=0, train_only=False,
                        show_plot=True, check_splits=True, stratify=True, random_state=23, shuffle=True)

    # 定义数据变换与模型
    tfms = [None, TSClassification()]
    batch_tfms = TSStandardize(by_sample=True)
    model = TSClassifier(X, y.values, splits=splits, path='models', arch=HybridHCAN,
                         tfms=tfms, batch_tfms=batch_tfms, metrics=accuracy)

    # 训练模型
    start_time = time.time()
    print("GPU available:", torch.cuda.is_available())
    model.fit_one_cycle(30, 0.001)
    end_time = time.time()
    print(f"训练时间：{end_time - start_time:.2f} 秒")
    model.export("stage1.pkl")
    print("模型结构：")
    print(model.model)

    # 加载测试数据
    X2, y2 = df2xy(pd.concat([df1[176000:], df2[176000:], df3[176000:]], ignore_index=True), data_cols=None)
    X2 = X2.reshape(264, 500, 3)
    X2 = np.transpose(X2, (0, 2, 1))
    label2 = [0] * 88 + [1] * 88 + [2] * 88
    y2 = pd.DataFrame(label2)

    from tsai.inference import load_learner
    mv_clf = load_learner(r'.\models\stage1.pkl')

    start_time1 = time.time()
    probas, target, preds = model.get_X_preds(X2[:264], y2.values[:264])
    end_time1 = time.time()
    print(f"推理时间：{end_time1 - start_time1:.2f} 秒")

    print(f'accuracy: {skm.accuracy_score(target.to("cpu").numpy().astype(int), preds.astype(int)):10.6f}')
    print(
        f'precision: {skm.precision_score(target.to("cpu").numpy().astype(int), preds.astype(int), average="weighted"):10.6f}')
    print(
        f'recall: {skm.recall_score(target.to("cpu").numpy().astype(int), preds.astype(int), average="weighted"):10.6f}')
    print(f'f1: {skm.f1_score(target.to("cpu").numpy().astype(int), preds.astype(int), average="weighted"):10.6f}')

    # 保存模型并统计文件大小与参数量
    with open(r'.\models\stage1.pkl', 'wb') as f:
        pickle.dump(model, f)
    file_size = os.path.getsize(r'.\models\stage1.pkl')
    print(f"模型文件大小：{file_size / (1024 * 1024):.2f} MB")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    params_count = count_parameters(model)
    print(f"模型参数数量：{params_count // 1000}K")

    # 展示结果与混淆矩阵
    model.show_results()
    cm = confusion_matrix(target.to("cpu").numpy().astype(int), preds.astype(int))
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 18})
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.show()

    # 可视化特征重要性与概率分布（根据需求添加相关代码）
    model.feature_importance()
    model.show_probas()

    # 此处可以添加更多绘图代码，如效率比较等
    # ...


if __name__ == '__main__':
    main()
