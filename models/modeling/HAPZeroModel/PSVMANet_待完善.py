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
from ..modeling import VisionTransformer, CONFIGS
import numpy as np
from torch.autograd import Variable
import torch.nn.init as init
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
        self.hidden_dim= np//ng 
        self.fc1 = nn.Linear(np, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, np)
        self.avgpool = nn.AdaptiveMaxPool1d(1)
        self.act=nn.GELU()

    def forward(self, x):#x(32,85,768)
        x_1 = self.fc1(self.avgpool(x).flatten(1)) #x_1(32,7)
        x_1 = self.act(x_1)#x_1(32,7)
        x_1 = F.sigmoid(self.fc2(x_1)).unsqueeze(-1)#x_1(32,7)->x_1(32,85,1)
        x_1 = x_1*x + x#(32,85,768)
        return x_1

class Tokenmix(nn.Module):
    def __init__(self, np):
        super(Tokenmix, self).__init__()
        dim = 784
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
        redisual =  x
        x=self.norm(x)
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
    def forward(self, q=None, k=None, v=None):#q(32,312,768) k(32,196,768) v(32,196,768)
        q, k, v = self.get_qkv(q, k, v)#都各自乘上W-> q(32,312,768) k(32,196,768) v(32,196,768)
        attn = torch.einsum("b q c, b k c -> b q k", q, k)#q(32,312,768) k(32,196,768)->(32,312,196)
        attn = self.act(attn)#(32,312,196)
        attn *= self.scale#(32,312,196)
        attn_mask = F.softmax(attn, dim=-1)#(32,312,196)
        out = torch.einsum("b q k, b k c -> b q c", attn_mask, v.float())#attn_mask(32,312,196)*v(32,196,768)->(32,312,768)
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



class Block(nn.Module):
    def __init__(self, dim, ffn_exp=4, drop_path=0.1, num_heads=1, num_parts=0,num_g=6):
        super(Block, self).__init__()
        self.dec_attn = AnyAttention(dim, True)
        self.ffn1 = Mlp(dim, hidden_features=dim * ffn_exp, act_layer=nn.GELU, norm_layer=Norm)
        self.drop_path = nn.Identity()
        self.reason = Tokenmix(dim)
        self.enc_attn = AnyAttention(dim, True)
        self.group_compact = SimpleReasoning(num_parts,num_g)
        self.maxpool1d = nn.AdaptiveMaxPool1d(1)
        self.enc_ffn = Mlp(dim, hidden_features=dim, act_layer=nn.GELU) 


    def forward(self, x, parts=None):#x(32,768,196) parts(32,85,768)
        x = rearrange(x, "b c p -> b p c")#(32,768,196)->(32,196,768)
        attn_0,attn_out = self.enc_attn(q=parts, k=x, v=x)#(32,85,768) (32,85,768)注意768是patch_dim
        attn_0= self.maxpool1d(attn_0).flatten(1)#(32,85,196)->(32,85,1)->(32,85)这里的物理意义在于1个属性对应196个patch的分数，maxpool就是计算1个属性最高分patch
        parts1 = parts + attn_out#残差连接,原先的属性原型向量被实例化更改了(32,85,768)，对应的是公式(3)
        parts2 = self.group_compact(parts1)#对应公式（4）和(5)->(32,85,768)
        if self.enc_ffn is not None:
            parts_out = parts2 + self.enc_ffn(parts2) + parts1#对应mlp ,即公式6(32,85,768)
            #parts_out = parts1#对应mlp ,即公式6(32,85,768)
        parts_d= parts+parts_out#对应公式6->(32,85,768)即针对每个样本的新的属性原型
        attn_1,attn_out = self.enc_attn(q=parts_d, k=x, v=x)#(32,85,196) (32,85,768)
        attn_1 = self.maxpool1d(attn_1).flatten(1)#(32,85)
        parts1_d = parts_d + attn_out#(32,312,768)
        parts_comp = self.group_compact(parts1_d)#(32,85,768)
        if self.enc_ffn is not None:
            parts_in = parts_comp + self.enc_ffn(parts_comp) +parts1_d#同上一样(32,85,768)
            #parts_in = parts1_d#同上一样(32,85,768)
        attn_mask, feats = self.dec_attn(q=x, k=parts_in, v=parts_in)#(32,196,85),(32,196,768)
        feats = x + feats#残差连接(32,196,768)
        feats = self.reason(feats)#(32,196,768)
        feats = feats + self.ffn1(feats)#(32,196,768)
        feats = rearrange(feats, "b p c -> b c p")#(32,196,768)->(32,768,196)
        return feats,attn_0,attn_1

class PSVMANet(nn.Module):
    def __init__(self, basenet, c,
                 attritube_num, cls_num, ucls_num, group_num, w2v,
                 scale=20.0, device=None):

        super(PSVMANet, self).__init__()
        self.attritube_num = attritube_num#312
        self.group_num=group_num#28
        self.feat_channel = c#768
        self.batch =10
        self.cls_num= cls_num#200
        self.ucls_num = ucls_num#50
        self.scls_num = cls_num - ucls_num#150
        if self.attritube_num == 85:
            #self.w2v_att = w2v[24:, :].float().to(device)
            self.w2v_att = w2v.float().to(device)
        else:
            self.w2v_att = torch.from_numpy(w2v).float().to(device)#(312,300)
        self.W = nn.Parameter(trunc_normal_(torch.empty(self.w2v_att.shape[1], self.feat_channel)),#(300,768)
                              requires_grad=True)#
        self.V = nn.Parameter(trunc_normal_(torch.empty(self.feat_channel, self.attritube_num)),#(768,312)
                              requires_grad=True)#
        assert self.w2v_att.shape[0] == self.attritube_num#
        if scale<=0:
            self.scale = nn.Parameter(torch.ones(1) * 20.0)
        else:
            self.scale = nn.Parameter(torch.tensor(scale), requires_grad=False)#

        #self.backbone_patch = nn.Sequential(*list(basenet.children()))[0]#PatchEmbed((proj): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))(norm): Identity())
        # self.backbone_drop= nn.Sequential(*list(basenet.children()))[1]
        # self.backbone_0 = nn.Sequential(*list(basenet.children()))[2][:-1]#VIT结构
        # self.backbone_1 = nn.Sequential(*list(basenet.children()))[2][-1]#VIT最后的Block
        # self.backbone_0 = nn.Sequential(*list(basenet.children()))[1][:-1]  # VIT结构
        # self.backbone_1 = nn.Sequential(*list(basenet.children()))[1][-1]  # VIT最后的Block
        self.basenet = basenet
        self.drop_path = 0.1

        # self.cls_token = basenet.cls_token#(1,1,768)
        # self.pos_embed = basenet.pos_embed#(1,196+1,768)

        self.cat = nn.Linear(self.attritube_num*self.feat_channel, attritube_num)#Linear(in_features=239616, out_features=312, bias=True)
        self.avgpool1d = nn.AdaptiveAvgPool1d(1)#AdaptiveAvgPool1d(output_size=1)
        self.CLS_loss = nn.CrossEntropyLoss()#
        self.Reg_loss = nn.MSELoss()

        # self.blocks = Block(self.feat_channel,
        #           num_heads=1,
        #           num_parts=self.attritube_num,
        #           num_g=self.group_num,
        #           ffn_exp=4,
        #           drop_path=0.1)
        # self.log_softmax_func = nn.LogSoftmax(dim=1)

        decoder_layer = TransformerDecoderLayer(d_model=300,
                                                nhead=1,
                                                dim_feedforward=2048,
                                                dropout=0.1,
                                                SAtt=True)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=1)
        self.W_1_s2v = nn.Parameter(nn.init.normal_(torch.empty(
            300, 300)), requires_grad=True)  #
        self.log_softmax_func = nn.LogSoftmax(dim=1)
        self.embed_cv = nn.Sequential(nn.Linear(768, 300))
        # self.bias = nn.Parameter(torch.tensor(1), requires_grad=False)
        # mask_bias = np.ones((1, 200))  # (1,200)[[1.,1.,1.....]]
        # mask_bias[:, self.seenclass.cpu().numpy()] *= -1  # (1，200),可见类的索引对应值标-1，不可见类索引对应的值标1
        # self.mask_bias = nn.Parameter(torch.tensor(
        #     mask_bias, dtype=torch.float), requires_grad=False)  # 类的索引对应值设置不梯度更新
    def compute_score(self, gs_feat,seen_att,att_all):
        gs_feat = gs_feat.view(self.batch, -1)
        gs_feat_norm = torch.norm(gs_feat, p=2, dim = 1).unsqueeze(1).expand_as(gs_feat)
        gs_feat_normalized = gs_feat.div(gs_feat_norm + 1e-5)
        temp_norm = torch.norm(att_all, p=2, dim=1).unsqueeze(1).expand_as(att_all)
        seen_att_normalized = att_all.div(temp_norm + 1e-5)
        score_o = torch.einsum('bd,nd->bn', gs_feat_normalized, seen_att_normalized)
        d, _ = seen_att.shape
        score_o = score_o*self.scale
        if d == self.cls_num:
            score = score_o
        if d == self.scls_num:
            score = score_o[:, :d]
            uu = self.ucls_num
            if self.training:
                mean1 = score_o[:, :d].mean(1)
                std1 = score_o[:, :d].std(1)
                mean2 = score_o[:, -uu:].mean(1)
                std2 = score_o[:, -uu:].std(1)
                mean_score = F.relu6(mean1 - mean2)
                std_score = F.relu6(std1 - std2)
                mean_loss = mean_score.mean(0) + std_score.mean(0)
                return score, mean_loss
        if d == self.ucls_num:
            score = score_o[:, -d:]
        return score,_
    # def compute_aug_cross_entropy(self, S_pp, Labels):
    #     # Labels = in_package['batch_label']
    #     # S_pp = in_package['pred']
    #
    #     # if self.is_bias:
    #     #     S_pp = S_pp - self.vec_bias
    #
    #     # if not self.is_conservative:
    #     #     S_pp = S_pp[:, self.seenclass]
    #     #     Labels = Labels[:, self.seenclass]
    #     #     assert S_pp.size(1) == len(self.seenclass)
    #
    #     Prob = self.log_softmax_func(S_pp)
    #     labels = torch.nn.functional.one_hot(Labels, num_classes=90).float()
    #     # if self.trian_class_counts != None:
    #     # batch_class_count = torch.matmul(Labels, self.trian_class_counts)
    #     # class_weights = (1 - 0.99) / (1 - 0.99 ** batch_class_count)
    #     loss = -torch.einsum('bk,bk->b', Prob, labels) #* (1 + class_weights)
    #     loss = torch.mean(loss)
    #     return loss

    def compute_aug_cross_entropy(self, S_pp, Labels
                                  ):
        # Labels = in_package['batch_label']
        # S_pp = in_package['pred']

        # if self.is_bias:
        #     S_pp = S_pp - self.vec_bias

        # if not self.is_conservative:
        #     S_pp = S_pp[:, self.seenclass]
        #     Labels = Labels[:, self.seenclass]
        #     assert S_pp.size(1) == len(self.seenclass)

        Prob = self.log_softmax_func(S_pp)
        labels = torch.nn.functional.one_hot(Labels, num_classes=90).float()
        # if self.trian_class_counts != None:
        #     batch_class_count = torch.matmul(Labels, self.trian_class_counts)
        #     class_weights = (1 - 0.99) / (1 - 0.99 ** batch_class_count)
        loss = -torch.einsum('bk,bk->b', Prob, labels) #* (1 + class_weights)
        loss = torch.mean(loss)
        return loss
    def forward_attribute(self, embed, att):  # (50,312)(B,a_num)
        embed = torch.einsum('ki,bi->bk', att, embed)  # self.att(200,312)连续值(class_num,a_num) embed(B,a_num)->(B,class_num)
        #self.vec_bias = self.mask_bias * self.bias  # (1,200)(1,class_num)*()->(1,200)(1,class_num)
        embed = embed #+ self.vec_bias  # (50,200)(B,class_num)+(1,200)->(50,200)(B,class_num)
        return embed
    def forward(self, x, att=None, label=None, seen_att=None, att_all=None):#x(b,c,h,w)(32,3,224,224)
        self.batch = x.shape[0]
        x, mask_x, attn_weights, mask_idxs = self.basenet(x)
        h_cv = self.embed_cv(x[:, 1:, :])
        #logits = self.head(x[:, 0])#(32,785,768)->(32,784,768)
        #parts = torch.einsum('lw,wv->lv', self.w2v_att, self.W)#(312,300)*(300,768)->(312,768)
        parts = self.w2v_att.expand(self.batch, -1, -1)#(312,768)->(32,312,768)
        #patches = self.backbone_patch(x)#(32,3,224,224)->(32,196,768)
        # patches = x#(32,3,224,224)->(32,196,768)由预训练的vit得到的patch特征
        #cls_token = self.cls_token.expand(x.shape[0], -1, -1) #(32,1,768) 表示属于类别的token
        # patches = torch.cat((cls_token, patches), dim=1) #(32,196,768)->(32,197,768)
        # feats_0 = self.backbone_drop(patches + self.pos_embed)#加上位置编码
        # feats_0 = self.backbone_0(feats_0)#输入给主vit网络->(32,197,768)
        # feats_in = x[:, 1:, :]#剔除bc的类别(32,197,768)->(32,196,768)
        # feats_out, att_0, att_1 = self.blocks(feats_in.transpose(1,2), parts=parts)#(32,768,196),(32,312),(32,312)
        # patches_1 = torch.cat((x[:, 0, :], feats_out.transpose(1,2)), dim=1) #(32,768,196)->(32,197,768)

        # feats_1 = x
        feats_1 = h_cv#(32,196,768)->(32,197,768)
        #feats_1, att_2, att_3 = self.blocks(feats_1.transpose(1, 2), parts=parts)#(32,768,196),(32,312),(32,312)

        feats = feats_1#(32,768,196)

        F_p_s2v = self.transformer_decoder(
            parts.permute(1, 0, 2), feats.permute(1, 0, 2))  # (50,312,300)(B,a_num,d_c),include eq14

        # out = self.avgpool1d(feats.view(self.batch, self.feat_channel, -1)).view(self.batch, -1)
        # out = torch.einsum('bc,cd->bd', out, self.V)
        S_p_s2v = torch.einsum('biv,vc,bic->bi', parts, self.W_1_s2v, F_p_s2v.permute(1, 0, 2))#eq15,(B,a_num,d_c)*(d_c,d_c)*(B,a_num,d_c)???->(B,a_num)
        #pred = self.forward_attribute(S_p_s2v, att_all)
        pred, b = self.compute_score(S_p_s2v, seen_att, att_all)
        # S_p_s2v_1 = torch.einsum('biv,vc,bic->bi', V_n_batch[:, :8, :], self.W_1_s2v_1, F_p_s2v_1)
        # S_p_s2v = torch.einsum('biv,vc,bic->bi', parts, self.W_1_s2v, F_p_s2v.permute(1, 0, 2))#eq15,(B,a_num,d_c)*(d_c,d_c)*(B,a_num,d_c)???->(B,a_num)
        # pred = self.forward_attribute(S_p_s2v, att_all)
        if not self.training:
            return pred

        #Lreg1 = self.Reg_loss(att_0, att)+self.Reg_loss(att_1, att)+self.Reg_loss(att_2, att)+self.Reg_loss(att_3, att)
        #Lreg1 = self.Reg_loss(att_2, att)+self.Reg_loss(att_3, att)
        #Lcls = self.CLS_loss(score, label)

        Lcls = self.compute_aug_cross_entropy(pred, label)
        scale = self.scale.item()
        loss_dict = {
            'Cls_loss': Lcls,
            'scale': scale,
            'bias_loss': b
        }

        return loss_dict

class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", SAtt=True):
        super(TransformerDecoderLayer, self).__init__(d_model, nhead,
                                                      dim_feedforward=dim_feedforward,
                                                      dropout=dropout,
                                                      activation=activation)
        self.SAtt = SAtt

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):  # tgt(312,50,300)(a_num,B,d_c) memory(196,50,300)(grid_num,B,d_c)
        if self.SAtt:  # True
            tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]  # tgt2(a_num,B,d_c) eq6-7
            tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[

            0]  # tgt)(a_num,B,d_c) memory(grid_num,B,d_c) memory(grid_num,B,d_c)->(312,50,300)(a_num,B,d_c)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
def build_PSVMANet(cfg):
    dataset_name = cfg.DATASETS.NAME
    info = utils.get_attributes_info(dataset_name)
    attritube_num = info["input_dim"]
    cls_num = info["n"]
    ucls_num = info["m"]
    group_num = info["g"] 
    c,w,h = 768, 28, 28
    scale = cfg.MODEL.SCALE
    #vit_model = create_model(num_classes=-1)
    # vit_model = create_model(num_classes=-1)
    # Prepare model
    config = CONFIGS['ViT-B_16']
    # if args.feature_fusion:
    #     config.feature_fusion = True
    # config.num_parts = args.num_parts
    # vit_model = VisionTransformer(config, cfg.DATASETS.IMAGE_SIZE, zero_head=True, num_classes=102, vis=True, smoothing_value=0.0, dataset='ip102')
    vit_model = VisionTransformer(config, cfg.DATASETS.IMAGE_SIZE,  num_classes=102, vis=True,smoothing_value=0.0, dataset='ip102')
    #vit_model.load_from(np.load('/home/star/jqwconda/TransFG-master/imagenet21k_ViT-B_16.npz'))
    # vit_model_path = "./pretrain_model_vit/vit_base_patch16_224.pth"
    weights_dict = np.load('/home/star/jqwconda/TransFG-master/imagenet21k_ViT-B_16.npz')
    # del_keys = ['head.weight', 'head.bias']# if vit_model.has_logits \
    # #     else ['head.weight', 'head.bias']
    # new_data = dict(weights_dict)
    # del new_data['weights_head']
    # for k in del_keys:
    #     del weights_dict[k]
    vit_model.load_state_dict(weights_dict, strict=False)
    w2v_file = dataset_name+"_attribute.pkl"
    w2v_path = join(cfg.MODEL.ATTENTION.W2V_PATH, w2v_file)

    with open(w2v_path, 'rb') as f:
        w2v = pickle.load(f)

    device = torch.device(cfg.MODEL.DEVICE)

    return PSVMANet(basenet=vit_model,
                  c=c,scale=scale,
                  attritube_num=attritube_num,
                  group_num=group_num, w2v=w2v,
                  cls_num=cls_num, ucls_num=ucls_num,
                  device=device)