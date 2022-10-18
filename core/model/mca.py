# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch, math
import scipy.io as io
import numpy as np
import os


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C, dim):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q_pos = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k_pos = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.img_box = nn.Linear(__C.HIDDEN_SIZE//__C.MULTI_HEAD, 1)
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        self.project = MLP(2*dim, __C.HIDDEN_SIZE, dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, k, q, mask, q_pos, k_pos, pos_scores, fg, box=None):
        n_batches = q.size(0)
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)
        # print(self.__C.MULTI_HEAD, self.__C.HIDDEN_SIZE_HEAD, v.size())
        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        if pos_scores is None and (fg == 1 or fg == 3):
            q_pos = self.linear_q_pos(q_pos).view(
                n_batches,
                -1,
                self.__C.MULTI_HEAD,
                self.__C.HIDDEN_SIZE_HEAD
            ).transpose(1, 2)

            k_pos = self.linear_k_pos(k_pos).view(
                n_batches,
                -1,
                self.__C.MULTI_HEAD,
                self.__C.HIDDEN_SIZE_HEAD
            ).transpose(1, 2)

        atted, pos_scores = self.att(v, k, q, mask, q_pos, k_pos, pos_scores, fg, box)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted, pos_scores

    def att(self, value, key, query, mask, q_pos, k_pos, pos_scores, fg, box):
        d_k = query.size(-1)
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)


        if pos_scores is None and (fg == 1 or fg == 3):
            pos_scores = torch.matmul(
                q_pos, k_pos.transpose(-2, -1)
            ) / math.sqrt(d_k)
        if fg == 1 or fg == 3:
            scores /= math.sqrt(2)
            pos_scores /= math.sqrt(2)

        if pos_scores is None and fg == 2:
            for i in range(self.__C.MULTI_HEAD):
                pos_emd = self.PositionalEmbedding(box, self.__C.HIDDEN_SIZE // self.__C.MULTI_HEAD)
                if i == 0:
                    pos_scores = self.relu(self.img_box(pos_emd)).squeeze().unsqueeze(1)
                else:
                    pos_scores = torch.cat((pos_scores, self.relu(self.img_box(pos_emd)).squeeze().unsqueeze(1)), 1)


        scores = scores + pos_scores

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value), pos_scores

    def PositionalEmbedding(self, f_g, dim_g, wave_len=1000):
        x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=-1)
        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5
        w = (x_max - x_min) + 1.
        h = (y_max - y_min) + 1.
        delta_x = cx - cx.view(cx.size()[0], 1, -1)
        delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
        delta_x = torch.log(delta_x)

        delta_y = cy - cy.view(cy.size()[0], 1, -1)
        delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
        delta_y = torch.log(delta_y)

        delta_w = torch.log(w / w.view(w.size()[0], 1, -1))
        delta_h = torch.log(h / h.view(h.size()[0], 1, -1))
        size = delta_h.size()

        delta_x = delta_x.view(size[0], size[1], size[2], 1)
        delta_y = delta_y.view(size[0], size[1], size[2], 1)
        delta_w = delta_w.view(size[0], size[1], size[2], 1)
        delta_h = delta_h.view(size[0], size[1], size[2], 1)

        position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)
        # print(position_mat.size())

        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, 1, -1)
        position_mat = position_mat.view(size[0], size[1], size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(size[0], size[1], size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)

        return embedding


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()
        self.mhatt = MHAtt(__C, 14)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, x_mask, x_pos, pos_scores=None):
        tmp, pos_scores = self.mhatt(x, x, x, x_mask, x_pos, x_pos, pos_scores, fg=1)
        x = self.norm1(x + self.dropout1(
            tmp
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x, pos_scores


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()
        self.mhatt1 = MHAtt(__C, 100)
        self.mhatt2 = MHAtt(__C, 14)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask, x_pos, y_pos, self_pos=None, co_pos=None, box=None):
        tmp, self_pos = self.mhatt1(x, x, x, x_mask, x_pos, x_pos, self_pos, fg=2, box=box)
        x = self.norm1(x + self.dropout1(
            tmp
        ))
        tmp, co_pos = self.mhatt2(y, y, x, y_mask, x_pos, y_pos, co_pos, fg=3)
        x = self.norm2(x + self.dropout2(
            tmp
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x, self_pos, co_pos


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, __C):
        super(MCA_ED, self).__init__()
        self.__C = __C

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])

    def forward(self, x, y, x_mask, y_mask, x_pos, y_pos, box):
        # Get hidden vector

        for enc in self.enc_list:
            x, _ = enc(x, x_mask, x_pos)

        for dec in self.dec_list:
            y, _, _ = dec(y, x, y_mask, x_mask, y_pos, x_pos, box=box)

        return x, y
