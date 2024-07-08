from matplotlib.cbook import flatten
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import os
from torch.nn.functional import dropout
from models.modeling import utils
from models.modeling.backbone_vit.vit_model import vit_base_patch16_224 as create_model

from os.path import join
import pickle
import numpy as np
from torch.autograd import Variable
import torch.nn.init as init
from timm.models.layers import DropPath
Norm =nn.LayerNorm


def trunc_normal_(tensor, mean=0, std=.01):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

class SimpleReasoning(nn.Module):
    def __init__(self, np,ng):
        super(SimpleReasoning, self).__init__()
        self.hidden_dim = np//ng
        self.fc1 = nn.Linear(np, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, np)
        self.avgpool = nn.AdaptiveMaxPool1d(1)
        self.act = nn.GELU()

    def forward(self, x):#x(32,85,768)
        x_1 = self.fc1(self.avgpool(x).flatten(1)) #x_1(32,7)
        x_1 = self.act(x_1)#x_1(32,7)
        x_1 = F.sigmoid(self.fc2(x_1)).unsqueeze(-1)#x_1(32,7)->x_1(32,85,1)
        x_1 = x_1*x + x#(32,85,768)
        return x_1

class Tokenmix(nn.Module):
    def __init__(self, np):
        super(Tokenmix, self).__init__()
        dim =196
        hidden_dim = 512
        dropout = 0.
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.act=nn.GELU()
        self.norm = nn.LayerNorm(np)
        self.net = nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout))
    def forward(self, x):
        redisual = x
        x = self.norm(x)
        x = rearrange(x, "b p c -> b c p")
        x = self.net(x)
        x = rearrange(x, "b c p-> b p c")
        out = redisual+x
        return out


class AnyAttention(nn.Module):
    def __init__(self, dim, qkv_bias=False):
        super(AnyAttention, self).__init__()
        self.norm_q, self.norm_k, self.norm_v = Norm(dim), Norm(dim), Norm(dim)
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.scale = dim ** (-0.5)
        self.act=nn.ReLU6()
        self.proj = nn.Linear(dim, dim)
    def get_qkv(self, q, k, v):
        q, k, v = self.norm_q(q), self.norm_k(k), self.norm_v(v)
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        return q, k, v
    def forward(self, q=None, k=None, v=None):#q(32,85,768) k(32,196,768) v(32,196,768)
        q, k, v = self.get_qkv(q, k, v)#都各自乘上W-> q(32,85,768) k(32,196,768) v(32,196,768)
        attn = torch.einsum("b q c, b k c -> b q k", q, k)#q(32,85,768) k(32,196,768)->(32,85,196)
        attn = self.act(attn)#(32,312,196)
        attn *= self.scale#(32,312,196)
        attn_mask = F.softmax(attn, dim=-1)#(32,312,196)
        out = torch.einsum("b q k, b k c -> b q c", attn_mask, v.float())#attn_mask(32,85,196)*v(32,196,768)->(32,85,768)
        out = self.proj(out)
        return attn, out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(hidden_features) or in_features
        self.norm = norm_layer(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.fc1.weight)
        bound1 = 1 / math.sqrt(fan_in1)
        nn.init.uniform_(self.fc1.bias, -bound1, bound1)
        fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.fc2.weight)
        bound2 = 1 / math.sqrt(fan_in2)
        nn.init.uniform_(self.fc2.bias, -bound2, bound2)


class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PromptingBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention, Mlp_block=Mlp
                 , init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.attn1(self.norm11(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.mlp1(self.norm21(x)))
        return x

class Block(nn.Module):
    def __init__(self, dim, ffn_exp=4, drop_path=0.1, num_heads=1, num_parts=0, num_g=6):
        super(Block, self).__init__()
        self.dec_attn = AnyAttention(dim, True)

        self.ffn1 = Mlp(dim, hidden_features=dim * ffn_exp, act_layer=nn.GELU, norm_layer=Norm)
        self.drop_path = nn.Identity()
        self.reason = Tokenmix(dim)
        self.enc_attn = AnyAttention(dim, True)
        self.group_compact = SimpleReasoning(num_parts, num_g)
        self.maxpool1d = nn.AdaptiveMaxPool1d(1)
        self.enc_ffn = Mlp(dim, hidden_features=dim, act_layer=nn.GELU)


    def forward(self, x, parts=None):#x(32,768,196) parts(32,85,768)
        x = rearrange(x, "b c p -> b p c")#(32,768,196)->(32,196,768)
        attn_0, attn_out = self.enc_attn(q=parts, k=x, v=x)#(32,85,768) (32,85,768)注意768是patch_dim
        attn_0 = self.maxpool1d(attn_0).flatten(1)#(32,85,196)->(32,85,1)->(32,85)这里的物理意义在于1个属性对应196个patch的分数，maxpool就是计算1个属性最高分patch
        parts1 = parts + attn_out #残差连接,原先的属性原型向量被实例化更改了(32,85,768)，对应的是公式(3)
        #parts2 = self.group_compact(parts1) #对应公式（4）和(5)->(32,85,768)
        if self.enc_ffn is not None:#MLP
            #parts_out = parts2 + self.enc_ffn(parts2) #+ parts1#对应mlp ,即公式6(32,85,768)
            parts_out = parts1 + self.enc_ffn(parts1) #+ parts1#对应mlp ,即公式6(32,85,768)
            #parts_out = parts1#对应mlp ,即公式6(32,85,768)
        parts_d = parts + parts_out#对应公式6->(32,85,768)即针对每个样本的新的属性原型
        attn_1, attn_out = self.enc_attn(q=parts_d, k=x, v=x)#(32,85,196) (32,85,768)
        attn_1 = self.maxpool1d(attn_1).flatten(1)#(32,85)
        parts1_d = parts_d + attn_out#(32,312,768)
        #parts_comp = self.group_compact(parts1_d)#(32,85,768)
        if self.enc_ffn is not None:#MLP
            #parts_in = parts_comp + self.enc_ffn(parts_comp)# + parts1_d#同上一样(32,85,768)
            parts_in = parts1_d + self.enc_ffn(parts1_d)# + parts1_d#同上一样(32,85,768)
            #parts_in = parts1_d#同上一样(32,85,768)
        _, feats = self.dec_attn(q=x, k=parts_in, v=parts_in)#(32,196,85),(32,196,768)
        feats = x + feats#残差连接(32,196,768)
        feats = self.reason(feats)#(32,196,768)
        feats = feats + self.ffn1(feats)#(32,196,768)
        feats = rearrange(feats, "b p c -> b c p")#(32,196,768)->(32,768,196)
        return feats, attn_0, attn_1

class HAPZeroNet(nn.Module):
    def __init__(self, basenet, c,
                 attritube_num, cls_num, ucls_num, group_num, w2v,
                 scale=20.0, device=None):

        super(HAPZeroNet, self).__init__()
        self.attritube_num = attritube_num#312
        self.group_num=group_num#28
        self.feat_channel = c#768
        self.batch =10
        self.cls_num= cls_num#200
        self.ucls_num = ucls_num#50
        self.scls_num = cls_num - ucls_num#150
        # self.seenclass = seenclass  # 150个可见类的Id（150）
        # self.unseenclass = unseenclass  # 50个不可见类的Id
        if self.attritube_num == 85:
            #self.w2v_att = w2v[24:, :].float().to(device)
            self.w2v_att = w2v.float().to(device)
        else:
            self.w2v_att = torch.from_numpy(w2v).float().to(device)#(312,300)
        self.W = nn.Parameter(trunc_normal_(torch.empty(self.w2v_att.shape[1], self.feat_channel)),#(300,768)
                              requires_grad=True)#
        self.V = nn.Parameter(trunc_normal_(torch.empty(self.feat_channel, self.attritube_num)),#(768,85)
                              requires_grad=True)#
        # self.V_1 = nn.Parameter(trunc_normal_(torch.empty(self.feat_channel, 8)),  # (768,85)
        #                       requires_grad=True)  #
        # self.V_2 = nn.Parameter(trunc_normal_(torch.empty(self.feat_channel, 77)),  # (768,85)
        #                       requires_grad=True)  #
        # self.V_3 = nn.Parameter(trunc_normal_(torch.empty(self.feat_channel, 61)),  # (768,85)
        #                       requires_grad=True)  #

        assert self.w2v_att.shape[0] == self.attritube_num#
        if scale<=0:
            self.scale = nn.Parameter(torch.ones(1) * 20.0)
        else:
            self.scale = nn.Parameter(torch.tensor(scale), requires_grad=False)#

        self.backbone_patch = nn.Sequential(*list(basenet.children()))[0]#PatchEmbed((proj): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))(norm): Identity())
        self.backbone_drop= nn.Sequential(*list(basenet.children()))[1]
        self.backbone_0 = nn.Sequential(*list(basenet.children()))[2][:-1]#VIT结构
        self.backbone_1 = nn.Sequential(*list(basenet.children()))[2][-1]#VIT最后的Block

        self.drop_path = 0.4

        self.cls_token = basenet.cls_token#(1,1,768)
        self.pos_embed = basenet.pos_embed#(1,196+1,768)

        self.cat = nn.Linear(self.attritube_num*self.feat_channel, attritube_num)#Linear(in_features=239616, out_features=312, bias=True)
        self.avgpool1d = nn.AdaptiveAvgPool1d(1)#AdaptiveAvgPool1d(output_size=1)
        self.CLS_loss = nn.CrossEntropyLoss()#
        self.Reg_loss = nn.MSELoss()
        #
        # self.blocks1 = Block(self.feat_channel,
        #                      num_heads=1,
        #                      num_parts=self.attritube_num,
        #                      num_g=self.group_num,
        #                      ffn_exp=4,
        #                      drop_path=0.4)
        # self.blocks1 = Block(self.feat_channel,
        #           num_heads=1,
        #           num_parts=8,
        #           num_g=self.group_num,
        #           ffn_exp=4,
        #           drop_path=0.4)
        # self.blocks2 = Block(self.feat_channel,
        #                     num_heads=1,
        #                     num_parts=77,
        #                     num_g=self.group_num,
        #                     ffn_exp=4,
        #                     drop_path=0.4)
        self.blocks = Block(self.feat_channel,
                            num_heads=1,
                            num_parts=85,
                            num_g=self.group_num,
                            ffn_exp=4,
                            drop_path=0.4)
        self.promptingblock = PromptingBlock(self.feat_channel, num_heads=1, drop_path=0.1)
        self.promptingblock_2 = PromptingBlock(self.feat_channel, num_heads=1, drop_path=0.1)
        self.promptingblock_3 = PromptingBlock(self.feat_channel, num_heads=1, drop_path=0.1)
        self.log_softmax_func = nn.LogSoftmax(dim=1)
        # self.log_softmax_func = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.maxpool1d = nn.AdaptiveMaxPool1d(1)
        self.prototype_0 = nn.Linear(self.feat_channel, 8, bias=False)
        self.prototype_1 = nn.Linear(self.feat_channel, 16, bias=False)
        self.prototype_2 = nn.Linear(self.feat_channel, 61, bias=False)
        self.matrix_0 = nn.Linear(self.feat_channel, 8, bias=False)
        self.matrix_1 = nn.Linear(self.feat_channel, 16, bias=False)
        self.matrix_2 = nn.Linear(self.feat_channel, 61, bias=False)
        # self.prototype_0 = nn.Linear(768, 8, bias=False)
    def compute_score(self, gs_feat, seen_att, att_all):
        gs_feat = gs_feat.view(self.batch, -1)
        gs_feat_norm = torch.norm(gs_feat, p=2, dim=1).unsqueeze(1).expand_as(gs_feat)
        gs_feat_normalized = gs_feat.div(gs_feat_norm + 1e-5)
        temp_norm = torch.norm(att_all, p=2, dim=1).unsqueeze(1).expand_as(att_all)
        seen_att_normalized = att_all.div(temp_norm + 1e-5)
        score_o = torch.einsum('bd,nd->bn', gs_feat_normalized, seen_att_normalized)
        d, _ = seen_att.shape#d=90
        score_o = score_o*self.scale#(32,102)
        if d == self.cls_num:#gzsl测试阶段
            score = score_o
        if d == self.scls_num:#训练阶段
            score = score_o[:, :d]#取前90
            uu = self.ucls_num#12
            if self.training:
                mean1 = score_o[:, :d].mean(1)#对前90个数取均值
                std1 = score_o[:, :d].std(1)#对前90个数取方差
                mean2 = score_o[:, -uu:].mean(1)#对后12个数取均值
                std2 = score_o[:, -uu:].std(1)#对后12个数取方差
                mean_score = F.relu6(mean1 - mean2)
                std_score = F.relu6(std1 - std2)
                mean_loss = mean_score.mean(0) + std_score.mean(0)
                #return score, mean_loss
                return score_o, mean_loss#原作者返回的是可见类的分数，但是我改成了返回所有类的分数
        if d == self.ucls_num:#czsl测试阶段 ucls_num=12
            score = score_o[:, -d:]
        return score, _
        # return score_o

    def compute_loss_Self_Calibrate(self, S_pp,unseenclass):
        # S_pp = in_package['S_pp']
        Prob_all = F.softmax(S_pp, dim=-1)
        Prob_unseen = Prob_all[:, unseenclass]
        assert Prob_unseen.size(1) == len(unseenclass)
        mass_unseen = torch.sum(Prob_unseen, dim=1)
        loss_pmp = -torch.log(torch.mean(mass_unseen))
        return loss_pmp

    def compute_aug_cross_entropy(self, S_pp, Labels, trian_class_counts):#(B,att_num),(B,)
        Prob = self.log_softmax_func(S_pp)
        labels = torch.nn.functional.one_hot(Labels, num_classes=102).float()
        if trian_class_counts != None:
            batch_class_count = torch.matmul(labels[:, :90], trian_class_counts)
            class_weights = (1 - 0.99) / (1 - 0.99 ** batch_class_count)
        loss = -torch.einsum('bk,bk->b', Prob, labels) * (1 + class_weights)
        loss = torch.mean(loss)
        return loss

    def compute_softmax_2(self, S_pp, Labels, att_all):  # (B,att_num),(B,)
        labels = torch.nn.functional.one_hot(Labels, num_classes=102).float()
        tgt = torch.matmul(labels, att_all)
        pred_1 = self.softmax(S_pp[:, :8])
        pred_2 = self.softmax(S_pp[:, 8:24])
        # pred_3 = self.softmax(S_pp[:, 24:])
        #pred_3 = self.softmax(S_pp[:, 24:])
        labels_1 = torch.argmax(tgt[:, :8], dim=1)
        labels_2 = torch.argmax(tgt[:, 8:24], dim=1)

        # labels_1_one_hot = torch.nn.functional.one_hot(labels_1, num_classes=8).float()
        # labels_2_one_hot = torch.nn.functional.one_hot(labels_2, num_classes=16).float()
        #labels_3 = torch.argmax(tgt[:, 24:], dim=1)
        loss_ce_1 = self.CLS_loss(pred_1, labels_1)
        # loss_ce_1 = torch.sum(-labels_1_one_hot * F.log_softmax(pred_1, dim=-1), dim=-1).mean()
        # loss_ce_2 = torch.sum(-labels_2_one_hot * F.log_softmax(pred_2, dim=-1), dim=-1).mean()
        loss_ce_2 = self.CLS_loss(pred_2, labels_2)

        #loss_ce_3 = self.CLS_loss(pred_3, labels_3)

        # Prob_1 = -self.log_softmax_func(S_pp[:, :8]).sum()
        # Prob_2 = -self.log_softmax_func(S_pp[:, 8:24]).sum()
        # Prob_3 = -self.log_softmax_func(S_pp[:, 24:]).sum()

        # loss_reg_1 = F.mse_loss(S_pp[:, :8], tgt[:, :8], reduction='mean')
        # loss_reg_2 = F.mse_loss(S_pp[:, 8:24], tgt[:, 8:24], reduction='mean')
        # loss_reg_3 = F.mse_loss(S_pp[:, 24:], tgt[:, 24:], reduction='mean')

        loss_reg = 0.2 * loss_ce_1 + 0.8 * loss_ce_2
        return loss_reg


    def compute_softmax(self, S_pp, Labels, att_all):  # (B,att_num),(B,)
        labels = torch.nn.functional.one_hot(Labels, num_classes=102).float()
        tgt = torch.matmul(labels, att_all)
        pred_1 = self.softmax(S_pp[:, :8])
        pred_2 = self.softmax(S_pp[:, 8:24])
        # pred_3 = self.softmax(S_pp[:, 24:])
        #pred_3 = self.softmax(S_pp[:, 24:])
        labels_1 = torch.argmax(tgt[:, :8], dim=1)
        labels_2 = torch.argmax(tgt[:, 8:24], dim=1)
        # labels_1 = torch.argmax(tgt[:, :8], dim=1)
        # labels_2 = tgt[:, 8:24].softmax(dim=1)
        # labels_1_one_hot = torch.nn.functional.one_hot(labels_1, num_classes=8).float()
        # labels_2_one_hot = torch.nn.functional.one_hot(labels_2, num_classes=16).float()
        #labels_3 = torch.argmax(tgt[:, 24:], dim=1)
        loss_ce_1 = self.CLS_loss(pred_1, labels_1)
        # loss_ce_1 = torch.sum(-labels_1_one_hot * F.log_softmax(pred_1, dim=-1), dim=-1).mean()
        # loss_ce_2 = torch.sum(-labels_2_one_hot * F.log_softmax(pred_2, dim=-1), dim=-1).mean()
        loss_ce_2 = self.CLS_loss(pred_2, labels_2)

        #loss_ce_3 = self.CLS_loss(pred_3, labels_3)

        # Prob_1 = -self.log_softmax_func(S_pp[:, :8]).sum()
        # Prob_2 = -self.log_softmax_func(S_pp[:, 8:24]).sum()
        # Prob_3 = -self.log_softmax_func(S_pp[:, 24:]).sum()

        # loss_reg_1 = F.mse_loss(S_pp[:, :8], tgt[:, :8], reduction='mean')
        # loss_reg_2 = F.mse_loss(S_pp[:, 8:24], tgt[:, 8:24], reduction='mean')
        # loss_reg_3 = F.mse_loss(S_pp[:, 24:], tgt[:, 24:], reduction='mean')

        loss_reg = 0.1 * loss_ce_1 + 1.2 * loss_ce_2
        return loss_reg

    def con_loss(self, features, labels):#(B,att_num),(B,)
        # features = in_package['embed']
        # labels = torch.argmax(Labels, dim=1)
        B, _ = features.shape
        features = F.normalize(features)
        cos_matrix = features.mm(features.t())  # (B,B) 样本特征之间的相似度
        pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()  # (B,B) 正样本关系矩阵 元素为0或1
        neg_label_matrix = 1 - pos_label_matrix  # 负样本关系矩阵
        pos_cos_matrix = 1 - cos_matrix
        neg_cos_matrix = cos_matrix - 0.4
        neg_cos_matrix[neg_cos_matrix < 0] = 0
        loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
        loss /= (B * B)
        return loss

    def forward(self, x, att=None, label=None, seen_att=None, att_all=None, seenclass=None, unseenclass=None, trian_class_counts=None):#x(b,c,h,w)(32,3,224,224)
        self.batch = x.shape[0]
        parts = torch.einsum('lw,wv->lv', self.w2v_att, self.W)#(85,300)*(300,768)->(85,768)
        parts = parts.expand(self.batch, -1, -1)#(312,768)->(32,85,768)
        patches = self.backbone_patch(x)#(32,3,224,224)->(32,196,768)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) #(32,1,768) 表示属于类别的token
        patches = torch.cat((cls_token, patches), dim=1) #(32,196,768)->(32,197,768)
        feats_0 = self.backbone_drop(patches + self.pos_embed)#加上位置编码
        feats_0 = self.backbone_0(feats_0)#输入给主vit网络->(32,197,768) 这部分及以上主要就是为图像生成VIT的特征

        #feats_in = feats_0[:, 1:, :]#剔除bc的类别(32,197,768)->(32,196,768)
        feats_p_1 = torch.cat((feats_0, self.prototype_0.weight.expand(self.batch, -1, -1)), dim=1)
        feats_p_1 = self.promptingblock(feats_p_1)
        p_1_out = feats_p_1[:, -8:, :]#(B,8,768)
        proto_cls_prob_1 = p_1_out @ (self.matrix_0.weight.T)
        # p_1_out = self.maxpool1d(p_1_out).flatten(1)
        # parts_1 = self.maxpool1d(parts[:, :8, :]).flatten(1)
        p_1_out = torch.diagonal(proto_cls_prob_1, dim1=-2, dim2=-1)
        feats_no_p_1 = feats_p_1[:, :-8, :]
        feats_p_2 = torch.cat((feats_no_p_1, self.prototype_1.weight.expand(self.batch, -1, -1)), dim=1)
        feats_p_2 = self.promptingblock_2(feats_p_2)
        p_2_out = feats_p_2[:, -16:, :]
        # p_2_out = self.maxpool1d(p_2_out).flatten(1)
        # parts_2 = self.maxpool1d(parts[:, 8:24, :]).flatten(1)
        proto_cls_prob_2 = p_2_out @ (self.matrix_1.weight.T)
        p_2_out = torch.diagonal(proto_cls_prob_2, dim1=-2, dim2=-1)
        feats_no_p_2 = feats_p_2[:, :-16, :]
        # feats_p_3 = torch.cat((feats_no_p_2, self.prototype_2.weight.expand(self.batch, -1, -1)), dim=1)
        # feats_p_3 = self.promptingblock_3(feats_p_3)
        # feats_no_p_3 = feats_p_3[:, :-61, :]
        # proto_cls_prob_3 = feats_no_p_3 @ (self.matrix_2.weight.T)
        # p_3_out = torch.diagonal(proto_cls_prob_3, dim1=-2, dim2=-1)

        feats_in = feats_no_p_2[:, 1:, :]#剔除bc的类别(32,197,768)->(32,196,768)
        p_out = torch.cat((p_1_out, p_2_out), dim=1)
        # p_out = torch.cat((p_out_1_2, p_3_out), dim=1)
        # feats_p_3 = torch.cat((feats_no_p_2, parts[:, 24:, :]), dim=1)

        ################################################################################################
        # 第一层-1
        feats_out, _, _ = self.blocks(feats_in.transpose(1, 2), parts=parts)  # (32,768,196),(32,85),(32,85)
        patches_1 = torch.cat((cls_token, feats_out.transpose(1, 2)), dim=1)  # (32,768,196)->(32,197,768)
        feats_1 = self.backbone_1(patches_1 + self.pos_embed)  # (32,197,768) 将结果输入给分类头
        feats_1 = feats_1[:, 1:, :]  # (32,196,768)->(32,197,768)# 剔除分类头结果
        # 第一层-2
        feats_1, attn_0, _ = self.blocks(feats_1.transpose(1, 2), parts=parts)  # (32,768,196),(32,85),(32,85)
        feats_1_ = feats_1  # (32,768,196)
        # 平均池化
        out_1 = self.avgpool1d(feats_1_.view(self.batch, self.feat_channel, -1)).view(self.batch, -1)
        # V_1 = self.V[:, :8]
        out = torch.einsum('bc,cd->bd', out_1, self.V)

        f_out_1 = (out[:, :8] + p_out[:, :8])/2
        f_out_2 = (out[:, 8:24] + p_out[:, 8:24])/2

        f_o = torch.cat((f_out_1, f_out_2), dim=1)
        # f_o = torch.cat((f_o_1, out[:, 24:]), dim=1)
        ################################################################################################
        # feats_out_2, _, _ = self.blocks2(feats_1, parts=parts_2)  # (32,768,196),(32,85),(32,85)
        # patches_2 = torch.cat((cls_token, feats_out_2.transpose(1, 2)), dim=1)  # (32,768,196)->(32,197,768)
        # feats_2 = self.backbone_1(patches_2 + self.pos_embed)  # (32,197,768) 将结果输入给分类头
        # feats_2 = feats_2[:, 1:, :]  # (32,196,768)->(32,197,768)# 剔除分类头结果
        # # 第一层-2
        # feats_2, _, _ = self.blocks2(feats_2.transpose(1, 2), parts=parts_2)  # (32,768,196),(32,85),(32,85)
        # feats_2_ = feats_2  # (32,768,196)
        # # 平均池化
        # out_2 = self.avgpool1d(feats_2_.view(self.batch, self.feat_channel, -1)).view(self.batch, -1)
        # # V_2 = self.V[:, 8:24]
        # out_2 = torch.einsum('bc,cd->bd', out_2, self.V_2)
        # out = torch.cat((out_1, out_2), dim=1)

        ################################################################################################
        # feats_out_3, _, _ = self.blocks3(feats_2, parts=parts_3)  # (32,768,196),(32,85),(32,85)
        # patches_3 = torch.cat((cls_token, feats_out_3.transpose(1, 2)), dim=1)  # (32,768,196)->(32,197,768)
        # feats_3 = self.backbone_1(patches_3 + self.pos_embed)  # (32,197,768) 将结果输入给分类头
        # feats_3 = feats_3[:, 1:, :]  # (32,196,768)->(32,197,768)# 剔除分类头结果
        # # 第一层-2
        # feats_3, _, _ = self.blocks3(feats_3.transpose(1, 2), parts=parts_3)  # (32,768,196),(32,85),(32,85)
        # feats_3_ = feats_3  # (32,768,196)
        # # 平均池化
        # out_3 = self.avgpool1d(feats_3_.view(self.batch, self.feat_channel, -1)).view(self.batch, -1)
        # # V_3 = self.V[:, 24:]
        # out_3 = torch.einsum('bc,cd->bd', out_3, self.V_3)
        #
        # out_1_2 = torch.cat((out_1, out_2), dim=1)
        # out = torch.cat((out_1_2, out_3), dim=1)

        # parts_1 = parts[:, :8, :]
        # parts_2 = parts[:, :24, :]
        # parts_3 = parts[:, :, :]
        # 解码器第一层
        # S_p_s2v_1 = self.decoderLayer1(feats_in, parts_1, cls_token, 0, 8)
        # S_p_s2v_2 = self.decoderLayer2(feats_in, parts_2, cls_token, 0, 24)
        # S_p_s2v_3 = self.decoderLayer3(feats_in, parts_3, cls_token, 0, 85)
        # B_att_1 = (S_p_s2v_1[:, :] + S_p_s2v_2[:, :8] + S_p_s2v_3[:, :8]) / 3
        # B_att_2 = (S_p_s2v_2[:, 8:24] + S_p_s2v_3[:, 8:24]) / 2
        # B_att_3 = S_p_s2v_3[:, 24:]
        # B_att_1_2 = torch.cat((B_att_1, B_att_2), dim=1)
        # B_att = torch.cat((B_att_1_2, B_att_3), dim=1)
        # pred_att = B_att
        #score, b = self.compute_score(out, seen_att, att_all)
        # 将属性向量转成分类结果，同时计算b损失
        score, b = self.compute_score(out, seen_att, att_all)#seen_att(90,85), att_all(102,85)

        if not self.training:
            return score


        #Lsa = self.compute_loss_Self_Calibrate(score, unseenclass)
        Lsa = torch.tensor(0)
        # Lreg = self.Reg_loss(att_0, att)+self.Reg_loss(att_1, att)+self.Reg_loss(att_2, att)+self.Reg_loss(att_3, att)
        # score_seen = score[:, seenclass]
        Lhp = self.compute_softmax(f_o, label, att_all)
        #Lhp = torch.tensor(0)
        #Lcls = self.CLS_loss(score, label)
        # Lcon = self.con_loss(out, label)
        Lce = self.compute_aug_cross_entropy(score, label, trian_class_counts)
        scale = self.scale.item()
        b = torch.tensor(0).to(x.device)
        loss_dict = {
            #'Reg_loss': Lreg,
            #'Con_loss': Lcon,
            'CE_loss': Lce,
            'HP_loss': Lhp,
            'SA_loss': Lsa,
            'scale': scale,
            'bias_loss': b
        }

        return loss_dict


def build_HAPZeroNet(cfg):
    dataset_name = cfg.DATASETS.NAME
    info = utils.get_attributes_info(dataset_name)
    attritube_num = info["input_dim"]
    cls_num = info["n"]
    ucls_num = info["m"]
    group_num = info["g"]
    c, w, h = 768, 14, 14
    scale = cfg.MODEL.SCALE
    vit_model = create_model(num_classes=-1)
    vit_model_path = "./pretrain_model_vit/vit_base_patch16_224.pth"
    weights_dict = torch.load(vit_model_path)
    del_keys = ['head.weight', 'head.bias'] if vit_model.has_logits \
        else ['head.weight', 'head.bias']
    for k in del_keys:
        del weights_dict[k]
    vit_model.load_state_dict(weights_dict, strict=False)
    w2v_file = dataset_name+"_attribute.pkl"
    w2v_path = join(cfg.MODEL.ATTENTION.W2V_PATH, w2v_file)

    with open(w2v_path, 'rb') as f:
        w2v = pickle.load(f)

    device = torch.device(cfg.MODEL.DEVICE)

    return HAPZeroNet(basenet=vit_model,
                  c=c,scale=scale,
                  attritube_num=attritube_num,
                  group_num=group_num, w2v=w2v,
                  cls_num=cls_num, ucls_num=ucls_num,
                  device=device)