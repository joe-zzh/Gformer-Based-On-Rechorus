# -*- coding: UTF-8 -*-

import os
import pickle
import argparse
import logging
import numpy as np
import pandas as pd

from utils import utils
from helpers.BaseReader import BaseReader

class GFormerReader(BaseReader):
    def __init__(self, args):
        super(GFormerReader, self).__init__(args)  # 调用父类构造函数
        self.norm_adj = None  # 在这里初始化 norm_adj，即归一化的邻接矩阵
        self.emb_size = 64  # 可根据需要修改嵌入维度
        self.n_layers = 2  # 可根据需要修改层数

        # 其他可能需要的初始化
        self._construct_norm_adj()

    def _construct_norm_adj(self):
        # 这里实现构建归一化邻接矩阵的逻辑
        # 例如，您可以创建一个基于用户和物品交互的邻接矩阵
        logging.info('Constructing normalized adjacency matrix...')
        # 假设，我们生成一个简单的邻接矩阵
        adjacency_matrix = np.zeros((self.n_users, self.n_items))
        for uid in self.train_clicked_set.keys():
            for iid in self.train_clicked_set[uid]:
                adjacency_matrix[uid][iid] = 1
        
        # 归一化处理，例如使用行归一化
        row_sum = adjacency_matrix.sum(axis=1, keepdims=True)
        self.norm_adj = adjacency_matrix / (row_sum + 1e-9)  # 加上一个小值避免除以零
        
        logging.info('Normalized adjacency matrix constructed.')

    @staticmethod
    def parse_data_args(parser):
        # 可以在此处添加 GFormerReader 特有的参数
        parser = BaseReader.parse_data_args(parser)
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Embedding size for the model.')
        parser.add_argument('--n_layers', type=int, default=2,
                            help='Number of layers in the model.')
        return parser

# 如果需要，可以在这里添加主函数示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    GFormerReader.parse_data_args(parser)
    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(level=logging.INFO)

    # 创建 GFormerReader 实例
    reader = GFormerReader(args)
