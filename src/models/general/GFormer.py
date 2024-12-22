# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

import torch
import numpy as np
import torch.nn as nn

from models.BaseModel import GeneralModel
class GFormer(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'n_layers', 'batch_size']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--n_layers', type=int, default=3,
                            help='Number of GFormer layers.')
        parser.add_argument('--selfloop_flag', type=bool, default=False,
                            help='Whether to add self-loop in adjacency matrix.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.user_num = corpus.n_users
        self.item_num = corpus.n_items
        self.norm_adj = self.build_adjmat(self.user_num, self.item_num, corpus.train_clicked_set, args.selfloop_flag)
        self.norm_adj_tensor = torch.tensor(self.norm_adj).float().to(self.device)  # 转换为适合的设备的张量
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(torch.empty(self.user_num, self.emb_size)),
            'item_emb': nn.Parameter(torch.empty(self.item_num, self.emb_size)),
        })
        nn.init.xavier_uniform_(self.embedding_dict['user_emb'])
        nn.init.xavier_uniform_(self.embedding_dict['item_emb'])
        self.layers = [self.emb_size] * self.n_layers

    def build_adjmat(self, user_count, item_count, train_mat, selfloop_flag=False):
        R = np.zeros((user_count, item_count), dtype=np.float32)
        for user in train_mat:
            for item in train_mat[user]:
                R[user, item] = 1.0

        adj_mat = np.zeros((user_count + item_count, user_count + item_count), dtype=np.float32)
        adj_mat[:user_count, user_count:] = R
        adj_mat[user_count:, :user_count] = R.T

        if selfloop_flag:
            np.fill_diagonal(adj_mat, 1)

        rowsum = np.array(adj_mat.sum(1)) + 1e-10
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)

        norm_adj_mat = d_mat_inv_sqrt @ adj_mat @ d_mat_inv_sqrt
        return norm_adj_mat  # 返回稠密矩阵

    def forward(self, feed_dict):
        user, items = feed_dict['user_id'], feed_dict['item_id']
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]

        # 遍历每一层
        for _ in range(self.n_layers):
            ego_embeddings = torch.mm(self.norm_adj_tensor, ego_embeddings)  # 使用稠密的 norm_adj 张量
            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_embeddings = all_embeddings[:self.user_num, :][user, :]
        item_embeddings = all_embeddings[self.user_num:, :][items, :]

        prediction = (user_embeddings[:, None, :] * item_embeddings).sum(dim=-1)  # [batch_size, -1]
        u_v = user_embeddings.unsqueeze(1).expand(-1, items.shape[1], -1)  # 避免调用 repeat 的性能损耗
        return {'prediction': prediction.view(feed_dict['batch_size'], -1), 'u_v': u_v, 'i_v': item_embeddings}

    def init_weights(self, m):
        if isinstance(m, nn.Parameter):
            nn.init.xavier_uniform_(m)

