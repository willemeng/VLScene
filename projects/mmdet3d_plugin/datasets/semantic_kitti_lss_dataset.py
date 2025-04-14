import copy
import tqdm
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.datasets.custom_3d import Custom3DDataset
import yaml

import mmcv
from os import path as osp
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .nuscnes_eval import NuScenesEval_custom
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
# from projects.mmdet3d_plugin.datasets.pipelines.loading import LoadOccupancy
from mmcv.parallel import DataContainer as DC
import random
import pdb, os
from glob import glob
# import glob
import numpy as np
from .semantic_kitti_dataset import CustomSemanticKITTIDataset

@DATASETS.register_module()
class CustomSemanticKITTILssDataset(CustomSemanticKITTIDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, random_camera=False, cbgs=False, repeat=1, corr='fog',severity_level='1',load_multi_voxel=False, *args, **kwargs):
        super(CustomSemanticKITTILssDataset, self).__init__(*args, **kwargs)
        
        self.random_camera = random_camera
        self.all_camera_ids = list(self.camera_map.values())
        self.load_multi_voxel = load_multi_voxel
        self.multi_scales = ["1_1", "1_2"]
        self.repeat = repeat
        self.cbgs = cbgs
        self.corr = corr
        self.severity_level = severity_level
        # print(self.corr+': '+self.severity_level)
        if self.repeat > 1:
            self.data_infos = self.data_infos * self.repeat
            random.shuffle(self.data_infos)
        # init class-balanced sampling
        self.data_infos = self.init_cbgs()

        # for lidar
        self.sizes = [256,256,32]
        self.lims = [[0,51.2],[-25.6,25.6],[-2,4.4]]
        self.data_config = yaml.safe_load(open("./semantickitti.yaml", 'r'))
        self.learning_map = self.data_config["learning_map"]
        self.learning_map_inv = self.data_config["learning_map_inv"]
        self.color_map = self.data_config['color_map']

        self._set_group_flag()

    @staticmethod
    def parse_poses(filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses
    def load_poses(self):
        """ read poses for each sequence

            Returns
            -------
            dict
                pose dict for different sequences.
        """
        pose_dict = dict()
        for sequence in self.sequences:
            pose_path = os.path.join(self.data_root, "dataset", "sequences", sequence, "poses.txt")
            calib = self.read_calib(
                os.path.join(self.data_root, "dataset", "sequences", sequence, "calib.txt")
            )
            pose_dict[sequence] = self.parse_poses(pose_path, calib)
        return pose_dict

    
    def prepare_cat_infos(self):
        tmp_file = 'semkitti_train_class_counts.npy'
        
        if not os.path.exists(tmp_file):
            class_counts_list = []
            for index in tqdm.trange(len(self)):
                info = self.data_infos[index]['voxel_path']
                assert info is not None    
                target_occ = np.load(info)
                
                # compute the class counts
                cls_ids, cls_counts = np.unique(target_occ, return_counts=True)
                class_counts = np.zeros(self.n_classes)
                
                cls_ids = cls_ids.astype(np.int)
                for cls_id, cls_count in zip(cls_ids, cls_counts):
                    # ignored
                    if cls_id == 255:
                        continue
                    
                    class_counts[cls_id] += cls_count
                
                class_counts_list.append(class_counts)
            
            # num_sample, num_class
            self.class_counts_list = np.stack(class_counts_list, axis=0)
            np.save(tmp_file, self.class_counts_list)
        else:
            self.class_counts_list = np.load(tmp_file)
    
    def init_cbgs(self):
        if not self.cbgs:
            return self.data_infos
        
        self.prepare_cat_infos()
        # remove unlabel class
        self.class_counts_list = self.class_counts_list[:, 1:]
        num_class = self.class_counts_list.shape[1]
        
        class_sum_counts = np.sum(self.class_counts_list, axis=0)
        sample_sum = class_sum_counts.sum()
        class_distribution = class_sum_counts / sample_sum
        
        # compute the balanced ratios
        frac = 1.0 / num_class
        ratios = frac / class_distribution
        ratios = np.log(1 + ratios)
        
        sampled_idxs_list = []
        for cls_id in range(num_class):
            # number of total points for this class
            num_class_sample_pts = class_sum_counts[cls_id] * ratios[cls_id]
            
            # get corresponding samples
            class_sample_valid_mask = (self.class_counts_list[:, cls_id] > 0)
            class_sample_valid_indices = class_sample_valid_mask.nonzero()[0]
            
            class_sample_points = self.class_counts_list[class_sample_valid_mask, cls_id]
            class_sample_prob = class_sample_points / class_sample_points.sum()
            class_sample_expectation = (class_sample_prob * class_sample_points).sum()
            
            # class_sample_mean = class_sample_points.mean()
            num_samples = int(num_class_sample_pts / class_sample_expectation)
            sampled_idxs = np.random.choice(class_sample_valid_indices, size=num_samples, p=class_sample_prob)
            sampled_idxs_list.extend(sampled_idxs)
        
        sampled_infos = [self.data_infos[i] for i in sampled_idxs_list]
        
        return sampled_infos

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        # example['lidar_data'] = input_dict['lidar_data']
        return example

    def get_ann_info(self, index):
        info = self.data_infos[index]['voxel_path']
        if info is None:
            return np.ones((256,256,32))

        if self.load_multi_voxel:
            annos = []
            for scale in self.multi_scales:
                scale_info = info.replace('1_1', scale)
                annos.append(np.load(scale_info))
            
            return annos
        else:
            gt = np.load(info)
            # if self.eval_range == 25.6:
            # gt[128:, :, :] = 255
            # gt[:, :64, :] = 255
            # gt[:, 192:, :] = 255

            # elif self.eval_range == 12.8:
            # gt[64:, :, :] = 255
            # gt[:, :96, :] = 255
            # gt[:, 160:, :] = 255

            # 25.6-51.2
            # gt[1:128, :, :] = 255
            # gt[:, 64:192, :] = 255
            return gt
        

    def get_data_info(self, index):
        info = self.data_infos[index]
        '''
        sample info includes the following:
            "img_2_path": img_2_path,
            "img_3_path": img_3_path,
            "sequence": sequence,
            "P2": P2,
            "P3": P3,
            "T_velo_2_cam": T_velo_2_cam,
            "proj_matrix_2": proj_matrix_2,
            "proj_matrix_3": proj_matrix_3,
            "voxel_path": voxel_path,
        '''
        
        input_dict = dict(
            occ_size = np.array(self.occ_size),
            pc_range = np.array(self.pc_range),
        )
        
        # load images, intrins, extrins, voxels
        image_paths = []
        lidar2cam_rts = []
        lidar2img_rts = []
        cam_intrinsics = []
        frame_id = info['img_2_path'].split('/')[-1][:-4]
        sequence = info['sequence']
        for cam_type in self.camera_used:
            if self.random_camera:
                cam_type = random.choice(self.all_camera_ids)

            image_paths.append(info['img_{}_path'.format(int(cam_type))])
      
            lidar2img_rts.append(info['proj_matrix_{}'.format(int(cam_type))])
      
            cam_intrinsics.append(info['P{}'.format(int(cam_type))])
       
            lidar2cam_rts.append(info['T_velo_2_cam'])
        

        # pose_list = self.poses[sequence]
        # seq_len = len(self.poses[sequence])
        # for i in self.target_frames:
        #     id = int(frame_id)

        #     if id + i < 0 or id + i > seq_len-1:
        #         target_id = frame_id
        #     else:
        #         target_id = str(id + i).zfill(6)

        #     rgb_path = os.path.join(
        #         self.data_root, "dataset", "sequences", sequence, "image_2", target_id + ".png"
        #     )

        #     pose_list = self.poses[sequence]

        #     ref = pose_list[int(frame_id)] # reference frame with GT semantic voxel
        #     target = pose_list[int(target_id)]
        #     ref2target = np.matmul(inv(target), ref) # both for lidar

        #     target2cam = scan["T_velo_2_cam"] # lidar to camera
        #     ref2cam = target2cam @ ref2target

        #     lidar2cam_rt  = ref2cam
        #     lidar2img_rt = (viewpad @ lidar2cam_rt)

        #     lidar2img_rts.append(lidar2img_rt)
        #     lidar2cam_rts.append(lidar2cam_rt)
        #     cam_intrinsics.append(intrinsic)
      
        calib_info = self.read_calib_file(info['calib_path'])
        calib = np.reshape(calib_info['P2'], [3, 4])[0, 0] * self.dynamic_baseline(calib_info)  
        # calib = np.reshape(calib_info['P2'], [3, 4])[0, 0] * 0.54

        input_dict.update(
            dict(
                img_filename=image_paths,    
       
                lidar2img=lidar2img_rts,
          
                cam_intrinsic=cam_intrinsics,
            
                lidar2cam=lidar2cam_rts,

                calib = calib,

                # lidar_data = data_collection
            ))
    
        # ground-truth in shape (256, 256, 32), XYZ order
        # TODO: how to do bird-eye-view augmentation for this? 
        input_dict['gt_occ'] = self.get_ann_info(index)
        if info['voxel_path'] is not None:
            target_1_2_path = info['voxel_path'].replace('1_1','1_2')
            # target_1_2_path = os.path.join(self.label_root, sequence, frame_id + "_1_2.npy")
            target_1_2 = np.load(target_1_2_path)
            target_1_2 = target_1_2.reshape(-1)
            target_1_2 = target_1_2.reshape(128, 128, 16)
            target_1_2 = target_1_2.astype(np.float32)
        else:
            target_1_2 = np.ones((128, 128, 16))
        input_dict['target_1_2'] = target_1_2
        return input_dict

    def get_remap_lut(self, completion=False):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = max(self.learning_map.keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(self.learning_map.keys())] = list(self.learning_map.values())

        # in completion we have to distinguish empty and invalid voxels.
        # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
        if completion:
            remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
            remap_lut[0] = 0  # only 'empty' stays 'empty'.
        
        return remap_lut
    def read_calib_file(self, filepath):
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data
    def dynamic_baseline(self, calib_info):
        P3 =np.reshape(calib_info['P3'], [3,4])
        P =np.reshape(calib_info['P2'], [3,4])
        baseline = P3[0,3]/(-P3[0,0]) - P[0,3]/(-P[0,0])
        return baseline

    def evaluate(self, results, logger=None, **kwargs):
        if 'ssc_scores' in results:
            ssc_scores = results['ssc_scores']
            
            class_ssc_iou = ssc_scores['iou_ssc'].tolist()
            res_dic = {
                "SC_Precision": ssc_scores['precision'].item(),
                "SC_Recall": ssc_scores['recall'].item(),
                "SC_IoU": ssc_scores['iou'],
                "SSC_mIoU": ssc_scores['iou_ssc_mean'],
            }
        else:
            assert 'ssc_results' in results
            ssc_results = results['ssc_results']
            completion_tp = sum([x[0] for x in ssc_results])
            completion_fp = sum([x[1] for x in ssc_results])
            completion_fn = sum([x[2] for x in ssc_results])
            
            tps = sum([x[3] for x in ssc_results])
            fps = sum([x[4] for x in ssc_results])
            fns = sum([x[5] for x in ssc_results])
            
            precision = completion_tp / (completion_tp + completion_fp)
            recall = completion_tp / (completion_tp + completion_fn)
            iou = completion_tp / \
                    (completion_tp + completion_fp + completion_fn)
            iou_ssc = tps / (tps + fps + fns + 1e-5)
            
            class_ssc_iou = iou_ssc.tolist()
            res_dic = {
                "SC_Precision": precision,
                "SC_Recall": recall,
                "SC_IoU": iou,
                "SSC_mIoU": iou_ssc[1:].mean(),
            }
        
        class_names = [
            'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
            'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
            'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
            'pole', 'traffic-sign'
        ]
        for name, iou in zip(class_names, class_ssc_iou):
            res_dic["SSC_{}_IoU".format(name)] = iou
        
        eval_results = {}
        for key, val in res_dic.items():
            eval_results['semkitti_{}'.format(key)] = round(val * 100, 2)
        
        # add two main metrics to serve as the sort metric
        eval_results['semkitti_combined_IoU'] = eval_results['semkitti_SC_IoU'] + eval_results['semkitti_SSC_mIoU']
        
        if logger is not None:
            logger.info('SemanticKITTI SSC Evaluation')
            logger.info(eval_results)
        
        return eval_results
        
def mask_op(data, x_min, x_max):
    mask = (data > x_min) & (data < x_max)
    return mask

def get_mask(pc, lims):
    mask_x = mask_op(pc[:, 0], lims[0][0] + 0.0001, lims[0][1] - 0.0001)
    mask_y = mask_op(pc[:, 1], lims[1][0] + 0.0001, lims[1][1] - 0.0001)
    mask_z = mask_op(pc[:, 2], lims[2][0] + 0.0001, lims[2][1] - 0.0001)
    mask = (mask_x) & (mask_y) & mask_z
    return mask
def unpack(compressed):
    ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1

    return uncompressed
def augmentation_random_flip(data, flip_type, is_scan=False):
    if flip_type==1:
        if is_scan:
            data[:, 0] = 51.2 - data[:, 0]
        else:
            data = np.flip(data, axis=0).copy()
    elif flip_type==2:
        if is_scan:
            data[:, 1] = -data[:, 1]
        else:
            data = np.flip(data, axis=1).copy()
    elif flip_type==3:
        if is_scan:
            data[:, 0] = 51.2 - data[:, 0]
            data[:, 1] = -data[:, 1]
        else:
            data = np.flip(np.flip(data, axis=0), axis=1).copy()
    return data
