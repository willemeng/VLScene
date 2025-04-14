# -*- coding:utf-8 -*-
# author: Xinge, Xzy
# @file: segmentator_3d_asymm_spconv.py

import numpy as np
import spconv.pytorch as spconv
import torch
from torch import nn
from projects.mmdet3d_plugin.occupancy.dense_heads.sdb import SDB
from projects.mmdet3d_plugin.occupancy.backbones.resnet3d import Bottleneck,BasicBlock7x7
from projects.mmdet3d_plugin.occupancy.image2bev.ViewTransformerLSSVoxel import *

def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                             padding=(0, 1, 1), bias=False, indice_key=indice_key)


def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride,
                             padding=(0, 0, 1), bias=False, indice_key=indice_key)


def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride,
                             padding=(0, 1, 0), bias=False, indice_key=indice_key)


def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
                             padding=(1, 0, 0), bias=False, indice_key=indice_key)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride,
                             padding=(1, 0, 1), bias=False, indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ResContextBlock, self).__init__()
        self.conv1 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.LeakyReLU()

        self.conv1_2 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.LeakyReLU()

        self.conv2 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        # shortcut.features = self.act1(shortcut.features)
        # shortcut.features = self.bn0(shortcut.features)
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))


        shortcut = self.conv1_2(shortcut)
        # shortcut.features = self.act1_2(shortcut.features)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))

        # shortcut.features = self.bn0_2(shortcut.features)
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        resA = self.conv2(x)
        # resA.features = self.act2(resA.features)
        resA = resA.replace_feature(self.act2(resA.features))

        # resA.features = self.bn1(resA.features)
        resA = resA.replace_feature(self.bn1(resA.features))

        resA = self.conv3(resA)
        # resA.features = self.act3(resA.features)
        resA = resA.replace_feature(self.act3(resA.features))
        # resA.features = self.bn2(resA.features)
        resA = resA.replace_feature(self.bn2(resA.features))

        # resA.features = resA.features + shortcut.features
        resA = resA.replace_feature(resA.features + shortcut.features)

        return resA


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3, 3), stride=1,
                 pooling=True, drop_out=True, height_pooling=False, indice_key=None):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        self.conv1 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act1 = nn.LeakyReLU()
        self.bn0 = nn.BatchNorm1d(out_filters)

        self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act1_2 = nn.LeakyReLU()
        self.bn0_2 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        if pooling:
            if height_pooling:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=2,
                                                padding=1, indice_key=indice_key, bias=False)
            else:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=(2, 2, 1),
                                                padding=1, indice_key=indice_key, bias=False)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        # shortcut.features = self.act1(shortcut.features)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))

        # shortcut.features = self.bn0(shortcut.features)
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        # shortcut.features = self.act1_2(shortcut.features)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))

        # shortcut.features = self.bn0_2(shortcut.features)
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        resA = self.conv2(x)
        # resA.features = self.act2(resA.features)
        resA = resA.replace_feature(self.act2(resA.features))
        
        # resA.features = self.bn1(resA.features)
        resA = resA.replace_feature(self.bn1(resA.features))

        resA = self.conv3(resA)
        # resA.features = self.act3(resA.features)
        resA = resA.replace_feature(self.act3(resA.features))

        # resA.features = self.bn2(resA.features)
        resA = resA.replace_feature(self.bn2(resA.features))

        # resA.features = resA.features + shortcut.features
        resA = resA.replace_feature(resA.features + shortcut.features)

        if self.pooling:
            resB = self.pool(resA)
            return resB, resA
        else:
            return resA


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None, up_key=None):
        super(UpBlock, self).__init__()
        # self.drop_out = drop_out
        self.trans_dilao = conv3x3(in_filters, out_filters, indice_key=indice_key + "new_up")
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.BatchNorm1d(out_filters)

        self.conv1 = conv1x3(out_filters, out_filters, indice_key=indice_key)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv1x3(out_filters, out_filters, indice_key=indice_key)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(out_filters)
        # self.dropout3 = nn.Dropout3d(p=dropout_rate)

        self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3, indice_key=up_key,
                                                  bias=False)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip):
        upA = self.trans_dilao(x)
        # upA.features = self.trans_act(upA.features)
        upA = upA.replace_feature(self.trans_act(upA.features))

        # upA.features = self.trans_bn(upA.features)
        upA = upA.replace_feature(self.trans_bn(upA.features))


        ## upsample
        upA = self.up_subm(upA)

        # upA.features = upA.features + skip.features
        upA = upA.replace_feature(upA.features + skip.features)

        upE = self.conv1(upA)
        # upE.features = self.act1(upE.features)
        upE = upE.replace_feature(self.act1(upE.features))

        # upE.features = self.bn1(upE.features)
        upE = upE.replace_feature(self.bn1(upE.features))

        upE = self.conv2(upE)
        # upE.features = self.act2(upE.features)
        upE = upE.replace_feature(self.act2(upE.features))

        # upE.features = self.bn2(upE.features)
        upE = upE.replace_feature(self.bn2(upE.features))

        upE = self.conv3(upE)
        # upE.features = self.act3(upE.features)
        upE = upE.replace_feature(self.act3(upE.features))
        
        # upE.features = self.bn3(upE.features)
        upE = upE.replace_feature(self.bn3(upE.features))

        return upE


class ReconBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ReconBlock, self).__init__()
        self.conv1 = conv3x1x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_3 = nn.BatchNorm1d(out_filters)
        self.act1_3 = nn.Sigmoid()

    def forward(self, x):
        shortcut = self.conv1(x)
        # shortcut.features = self.bn0(shortcut.features)
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        # shortcut.features = self.act1(shortcut.features)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))

        shortcut2 = self.conv1_2(x)
        # shortcut2.features = self.bn0_2(shortcut2.features)
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        # shortcut2.features = self.act1_2(shortcut2.features)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))

        shortcut3 = self.conv1_3(x)
        # shortcut3.features = self.bn0_3(shortcut3.features)
        shortcut = shortcut.replace_feature(self.bn0_3(shortcut.features))

        # shortcut3.features = self.act1_3(shortcut3.features)
        shortcut = shortcut.replace_feature(self.act1_3(shortcut.features))
        
        # shortcut.features = shortcut.features + shortcut2.features + shortcut3.features
        shortcut = shortcut.replace_feature(shortcut.features + shortcut2.features + shortcut3.features)

        # shortcut.features = shortcut.features * x.features
        shortcut = shortcut.replace_feature(shortcut.features * x.features)

        return shortcut


def extract_nonzero_features(x):
    device = x.device
    nonzero_index = torch.sum(torch.abs(x), dim=1).nonzero()
    coords = nonzero_index.type(torch.int32).to(device)
    channels = int(x.shape[1])
    features = x.permute(0, 2, 3, 4, 1).reshape(-1, channels)
    features = features[torch.sum(torch.abs(features), dim=1).nonzero(), :]
    features = features.squeeze(1).to(device)
    coords, _, _ = torch.unique(coords, return_inverse=True, return_counts=True, dim=0)
    return coords, features


class GeometricSemanticSparseAwareness(nn.Module):
    def __init__(self,
                 output_shape=[128,128,16],
                 num_input_features=128,
                 init_size=64):
        super(GeometricSemanticSparseAwareness, self).__init__()
        self.strict = False

        sparse_shape = np.array(output_shape)
        self.sparse_shape = sparse_shape

        self.pre_conv = nn.Sequential(  nn.Conv3d(num_input_features, init_size, kernel_size=(1, 1, 1),
                                        stride=(1, 1, 1), bias=False),
                                        # nn.BatchNorm3d(init_size),
                                        nn.ReLU(inplace=True),
                                        # BasicBlock7x7(init_size,init_size),
                                        BasicBlock7x7(init_size,init_size),
                                        )

        ### Segmentation sub-network
        self.downCntx = ResContextBlock(init_size, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2,height_pooling=True, indice_key="down3")

        self.upBlock2 = UpBlock(4 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 *init_size, 2 *init_size, indice_key="recon")



    def forward(self, x,train=True):
        # TODO:: exp3: use BasicBlock7x7

        # Neighborhood Geometry Propagation
        x = self.pre_conv(x)



        # Dense to sparse
        coord, features = extract_nonzero_features(x)
        x = spconv.SparseConvTensor(features, coord.int(), self.sparse_shape, x.shape[0])  # voxel features

        # Sparse Semantic Interaction.
        x = self.downCntx(x)
        down1c, down1b = self.resBlock2(x)
        down2c, down2b = self.resBlock3(down1c)

        up2e = self.upBlock2(down2c, down2b)
        up1e = self.upBlock3(up2e, down1b)
        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        y = up0e.dense()

        occ= None
        return y,occ
        
    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x