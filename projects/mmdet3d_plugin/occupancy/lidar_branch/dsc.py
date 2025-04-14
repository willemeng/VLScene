#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv

from .preprocess import PcPreprocessor
from .bev_net import BEVUNet, BEVUNetv1
from .completion import CompletionBranch
from .semantic_segmentation import SemanticBranch
from .lovasz_losses import lovasz_softmax

class DSC(nn.Module):
    def __init__(self, phase='trainval'):
        super().__init__()
        self.phase = phase
        nbr_classes =  [
    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
    'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
    'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
    'pole', 'traffic-sign'
]
        self.nbr_classes = len(nbr_classes)
        # self.class_frequencies = cfg['DATASET']['SC_CLASS_FREQ']
        # ss_req = cfg['DATASET']['SS_CLASS_FREQ']
        self.class_weights = torch.from_numpy(np.array([0.446, 0.603, 0.852, 0.856, 0.747, 0.734, 0.801, 0.796, 0.818, 0.557, 0.653, 0.568, 0.683, 0.560, 0.603, 0.530, 0.688, 0.574, 0.716, 0.786]))

        self.lims = [[0, 51.2], [-25.6, 25.6], [-2, 4.4]]
        self.sizes = [256, 256, 32]  # W, H, D (x, y, z)
        self.grid_meters = [0.2, 0.2, 0.2]
        self.n_height = self.sizes[-1] # 32
        self.dilation = 1
        self.bilinear = True
        self.group_conv = False
        self.input_batch_norm = True
        self.dropout = 0.5
        self.circular_padding = False
        self.dropblock = False
        self.preprocess = PcPreprocessor(lims=self.lims, sizes=self.sizes, grid_meters=self.grid_meters, init_size=self.n_height)
        self.sem_branch = SemanticBranch(sizes=self.sizes, nbr_class=20-1, init_size=self.n_height, phase=phase)
        self.com_branch = CompletionBranch(init_size=self.n_height, nbr_class=20, phase=phase)

        self.bev_model = BEVUNetv1(self.nbr_classes*self.n_height, self.n_height, self.dilation, self.bilinear, self.group_conv,
                            self.input_batch_norm, self.dropout, self.circular_padding, self.dropblock)

    def forward(self, example,image_voxel):
        batch_size = len(example['points'])
        with torch.no_grad():
            indicator = [0]
            pc_ibatch = []
            for i in range(batch_size):
                pc_i = example['points'][i]
                pc_ibatch.append(pc_i)
                indicator.append(pc_i.size(0) + indicator[-1])
            pc = torch.cat(pc_ibatch, dim=0)
        vw_feature, coord_ind, full_coord, info = self.preprocess(pc, indicator)  # N, C; B, C, W, H, D
        coord = torch.cat([coord_ind[:, 0].reshape(-1, 1), torch.flip(coord_ind, dims=[1])[:, :3]], dim=1)
        bev_dense = self.sem_branch.bev_projection(vw_feature, coord, np.array(self.sizes, np.int32)[::-1], batch_size) # B, C, H, W
        # voxel_dense = spconv.SparseConvTensor(
        #     vw_feature, coord.int(), (np.array(self.sizes, np.int32))[::-1], batch_size
        # ).dense()
        # torch.cuda.empty_cache()

        ss_data_dict = {}
        ss_data_dict['vw_features'] = vw_feature
        ss_data_dict['coord_ind'] = coord_ind
        ss_data_dict['full_coord'] = full_coord
        ss_data_dict['info'] = info
        
        ss_out_dict = self.sem_branch(ss_data_dict, example)  # B, C, D, H, W
        sc_data_dict = {}
        occupancy = example['occupancy'].permute(0, 3, 2, 1)# B, D, H, W

        sc_data_dict['vw_dense'] = occupancy.unsqueeze(1)
        sc_out_dict = self.com_branch(sc_data_dict, example)

        inputs = torch.cat([occupancy, bev_dense], dim=1)  # B, C, H, W
        
        x = self.bev_model(inputs, ss_out_dict['mss_bev_dense'], sc_out_dict['mss_bev_dense'],image_voxel)
        new_shape = [x.shape[0], self.nbr_classes, self.n_height, *x.shape[-2:]]    # [B, 20, 32, 256, 256]
        x = x.view(new_shape)
        out_scale_1_1 = x.permute(0,1,4,3,2)   # [B,20,256,256,32]
        if self.phase == 'trainval':
            loss_dict = self.compute_loss(out_scale_1_1, self.get_target(example)['1_1'], ss_out_dict['loss'], sc_out_dict['loss'])
            return {'output_voxels': out_scale_1_1}, loss_dict

        return {'output_voxels': out_scale_1_1}

    def compute_loss(self, scores, labels, ss_loss_dict, sc_loss_dict):
        '''
        :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
        '''
        class_weights = self.class_weights.to(device=scores.device, dtype=scores.dtype)

        loss_1_1 = F.cross_entropy(scores, labels.long(), weight=class_weights, ignore_index=255)
        loss_1_1 += lovasz_softmax(torch.nn.functional.softmax(scores, dim=1), labels.long(), ignore=255)
        loss_1_1 *= 3

        loss_seg = sum(ss_loss_dict.values())
        loss_com = sum(sc_loss_dict.values())
        loss_total = loss_1_1 + loss_seg + loss_com
        loss = {'loss_total':loss_total,'semantic_1_1': loss_1_1, 'semantic_seg': loss_seg, 'scene_completion': loss_com}

        return loss

    def weights_initializer(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def weights_init(self):
        self.apply(self.weights_initializer)

    def get_parameters(self):
        return self.parameters()

    def get_class_weights(self):
        '''
        Class weights being 1/log(fc) (https://arxiv.org/pdf/2008.10559.pdf)
        '''
        epsilon_w = 0.001  # eps to avoid zero division
        weights = torch.from_numpy(1 / np.log(np.array(self.class_frequencies) + epsilon_w))

        return weights

    def get_target(self, data):
        '''
        Return the target to use for evaluation of the model
        '''
        label_copy = data['label_1_1'].clone()
        label_copy[data['invalid_1_1'] == 1] = 255
        return {'1_1': label_copy}

    def get_scales(self):
        '''
        Return scales needed to train the model
        '''
        scales = ['1_1']
        return scales

    def get_validation_loss_keys(self):
        # return ['total','loss_ssc', 'loss_sem_scal','loss_geo_scal', 'semantic_1_1', 'semantic_seg', 'scene_completion']
        return ['total','semantic_1_1', 'semantic_seg', 'scene_completion']

    def get_train_loss_keys(self):
        # return ['total','loss_ssc', 'loss_sem_scal','loss_geo_scal', 'semantic_1_1', 'semantic_seg', 'scene_completion']
        return ['total', 'semantic_1_1', 'semantic_seg', 'scene_completion']

