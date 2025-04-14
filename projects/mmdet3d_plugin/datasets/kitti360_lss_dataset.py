import os
from os import path as osp
from PIL import Image
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.linalg import inv
from torchvision import transforms
from mmdet.datasets import DATASETS
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import Compose

@DATASETS.register_module()
class Kitti360Dataset(Dataset):
    CLASSES = ('empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'road',
         'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'terrain',
         'pole', 'traffic-sign', 'other-structure', 'other-object')
    def __init__(
        self,
        split,
        test_mode,
        data_root,
        ann_file,
        pipeline,
        classes,
        modality,
        box_type_3d,
        occ_size,
        pc_range,
        queue_length,
        camera_used,
        preprocess_root,
        img_size=[384, 1408],
        temporal = [-3,-6,-9,-12],
        eval_range = 51.2,
        depthmodel="msnet3d",
        labels_tag = 'labels',
        color_jitter=None,
        scale=2
    ):
        super().__init__()
        self.ann_file = data_root
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        self.camera_map = {'left': '00', 'right': '01'}
        self.camera_used = [self.camera_map[camera] for camera in camera_used]
        self.data_root = data_root
        self.label_root = os.path.join(preprocess_root, labels_tag)
        self.depth_query = "msnet3d"
        self.depthmodel = depthmodel
        self.eval_range = eval_range
        splits = {
            "train": ['2013_05_28_drive_0004_sync', '2013_05_28_drive_0000_sync', '2013_05_28_drive_0010_sync',
     '2013_05_28_drive_0002_sync', '2013_05_28_drive_0003_sync', '2013_05_28_drive_0005_sync',
     '2013_05_28_drive_0007_sync'],
            "val": ['2013_05_28_drive_0006_sync'],
            "test": ['2013_05_28_drive_0009_sync'],
        }
        self.split = split 
        self.sequences = splits[split]
        self.class_names =  ['empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'road',
         'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'terrain',
         'pole', 'traffic-sign', 'other-structure', 'other-object']

        self.scene_size = (51.2, 51.2, 6.4)
        self.vox_origin = np.array([0, -25.6, -2])
        self.voxel_size = 0.2  # 0.2m
        self.scale = scale

        self.img_W = img_size[1]
        self.img_H = img_size[0]

        # self.poses=self.load_poses()
        # self.target_frames = temporal
        self.load_scans()
        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )
        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.test_mode = test_mode
        self.set_group_flag()
        

    def __getitem__(self, index):
        
        return self.prepare_data(index)

    def __len__(self):
        return len(self.scans)

    @staticmethod
    def read_calib():
        P0 = np.array([
            552.554261,
            0.000000,
            682.049453,
            0.000000,
            0.000000,
            552.554261,
            238.769549,
            0.000000,
            0.000000,
            0.000000,
            1.000000,
            0.000000,
        ]).reshape(3, 4)

        P1 = np.array([
                552.554261,
                0.000000,
                682.049453,
                -328.318735,
                0.000000,
                552.554261,
                238.769549,
                0.000000,
                0.000000,
                0.000000,
                1.000000,
                0.000000,
            ]).reshape(3, 4)
        cam2velo = np.array([
            0.04307104361,
            -0.08829286498,
            0.995162929,
            0.8043914418,
            -0.999004371,
            0.007784614041,
            0.04392796942,
            0.2993489574,
            -0.01162548558,
            -0.9960641394,
            -0.08786966659,
            -0.1770225824,
        ]).reshape(3, 4)
        C2V = np.concatenate([cam2velo, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)
        V2C = np.linalg.inv(C2V)
        V2C = V2C[:3, :]

        out = {}
        out["P2"] = np.identity(4)
        out["P3"] = np.identity(4)
        out['P2'][:3, :4] = P0
        out['P3'][:3, :4] = P1
        out['Tr'] = np.identity(4)
        out['Tr'][:3, :4] = V2C
        return out

    @staticmethod
    def parse_poses(filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)
        match_path= filename.replace('poses','match')
        match = open(match_path)

        poses = dict()

        Tr = calibration["Tr"]
        Tr_inv = inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]
            pose = np.zeros((4, 4))
            index = int(values[0])
            pose[0, 0:4] = values[1:5]
            pose[1, 0:4] = values[5:9]
            pose[2, 0:4] = values[9:13]
            pose[3, 3] = 1.0
            pose = np.matmul(Tr_inv, np.matmul(pose, Tr))
            poses.update({str(index):pose})
            # poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
        match_poses = []
        for line in match:
            ind = int(line.strip().split()[-2][4:10])
            match_poses.append(poses[str(ind)])
        return match_poses

    def load_poses(self):
        """ read poses for each sequence

            Returns
            -------
            dict
                pose dict for different sequences.
        """
        pose_dict = dict()
        calib = self.read_calib()
        for sequence in self.sequences:
            pose_path = os.path.join(self.data_root, "data_2d_raw", sequence, "poses.txt")
            pose_dict[sequence] = self.parse_poses(pose_path, calib)
        return pose_dict

    def load_scans(self):
        """ read each scan

            Returns
            -------
            list
                list of each single scan.
        """
        self.scans = []
        calib = self.read_calib()
        for sequence in self.sequences:
            P2 = calib['P2']
            P3 = calib['P3']

            T_velo_2_cam = calib['Tr']
            proj_matrix_2 = P2 @ T_velo_2_cam
            proj_matrix_3 = P3 @ T_velo_2_cam

            glob_path = osp.join(self.data_root, 'data_2d_raw', sequence, 'voxels', '*.bin')
            for voxel_path in glob.glob(glob_path):
            
                self.scans.append({
                    'sequence': sequence,
                    # 'pose': self.poses[sequence],
                    'P0': P2,
                    'P1': P3,
                    'T_velo_2_cam': T_velo_2_cam,
                    'proj_matrix_0': proj_matrix_2,
                    'proj_matrix_1': proj_matrix_3,
                    'voxel_path': voxel_path,
                })

    def set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def prepare_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        data_queue = []
        example = self.get_data_info(index)

        data_queue.insert(0, example)

        return self.union2one(data_queue)

    def union2one(self, queue):
        """
        convert sample queue into one single sample.
        """
        # imgs_list = [each['img'] for each in queue]
        metas_map = {}

        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas']

        # queue[-1]['img'] = DC(torch.stack(imgs_list),
        #                       cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines.
        """
        scan = self.scans[index]

        voxel_path = scan["voxel_path"]

        sequence = scan["sequence"]
        filename = os.path.basename(voxel_path)
        frame_id = os.path.splitext(filename)[0]

        meta_dict = self.get_meta_info(scan, sequence, frame_id)
        target = self.get_gt_info(sequence, frame_id)
        input_dict = meta_dict
        input_dict['gt_occ'] = target
        input_dict['occ_size'] = [256, 256, 32]
        input_dict['pc_range'] = [0, -25.6, -2, 51.2, 25.6, 4.4]
        example = self.pipeline(input_dict)
        return example


    def get_meta_info(self, scan, sequence, frame_id):
        """Get meta info according to the given index.

        Args:
            scan (dict): scan information,
            sequence (str): sequence id,
            frame_id (str): frame id,

        Returns:
            dict: Meta information that will be passed to the data \
                preprocessing pipelines.
        """
        rgb_path = os.path.join(
            self.data_root, "data_2d_raw", sequence, 'image_00/data_rect', frame_id + ".png"
        )

        # for multiple images
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        image_paths = []
        # ref2targets = []
        image_paths.append(rgb_path)
        rgb_path = os.path.join(
            self.data_root, "data_2d_raw", sequence, 'image_01/data_rect', frame_id + ".png"
        )
        image_paths.append(rgb_path)

        for cam_type in self.camera_used:

            lidar2img_rts.append(scan['proj_matrix_{}'.format(int(cam_type))])
      
            cam_intrinsics.append(scan['P{}'.format(int(cam_type))])
       
            lidar2cam_rts.append(scan['T_velo_2_cam'])


        calib = np.reshape(scan['P0'][:3], [3, 4])[0, 0] * 0.6

        # pose_list = self.poses[sequence]
        # ref = pose_list[int(frame_id)] # reference frame with GT semantic voxel
        # target = pose_list[int(frame_id)]
        # ref2target = np.matmul(inv(target), ref) # both for lidar
        # ref2targets.append(ref2target)
        # ref2targets.append(ref2target)
        # seq_len = len(self.poses[sequence])
        # for i in self.target_frames:
        #     id = int(frame_id)

        #     if id + i < 0 or id + i > seq_len-1:
        #         target_id = frame_id
        #     else:
        #         target_id = str(id + i).zfill(6)

        #     rgb_path = os.path.join(
        #         self.data_root, "data_2d_raw", sequence, 'image_00/data_rect', target_id + ".png"
        #     )
        #     image_paths.append(rgb_path)
        #     lidar2img_rts.append(scan['proj_matrix_0'])
      
        #     cam_intrinsics.append(scan['P0'])
       
        #     lidar2cam_rts.append(scan['T_velo_2_cam'])

        #     pose_list = self.poses[sequence]

        #     ref = pose_list[int(frame_id)] # reference frame with GT semantic voxel
        #     target = pose_list[int(target_id)]
        #     ref2target = np.matmul(inv(target), ref) # both for lidar
        #     ref2targets.append(ref2target)
        target_1_2 = np.ones((128, 128, 16))
        meta_dict = dict(
                img_filename=image_paths,    
       
                lidar2img=lidar2img_rts,
          
                cam_intrinsic=cam_intrinsics,
            
                lidar2cam=lidar2cam_rts,

                calib = calib,

                target_1_2 = target_1_2,
                # ref2targets = ref2targets
        )

        return meta_dict

    def get_input_info(self, sequence, frame_id):
        """Get the image of the specific frame in a sequence.

        Args:
            sequence (str): sequence id,
            frame_id (str): frame id.

        Returns:
            torch.tensor: Img.
        """
        seq_len = 9#len(self.poses[sequence])
        image_list = []

        rgb_path = os.path.join(
            self.data_root, "data_2d_raw", sequence, 'image_00/data_rect', frame_id + ".png"
        )
        img = Image.open(rgb_path).convert("RGB")
        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)
        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0
        img = img[:self.img_H, :self.img_W, :]  # crop image
        image_list.append(self.normalize_rgb(img))

        # reference frame
        for i in self.target_frames:
            id = int(frame_id)

            if id + i < 0 or id + i > seq_len-1:
                target_id = frame_id
            else:
                target_id = str(id + i).zfill(6)

            if int(frame_id) not in self.poses[sequence] or int(target_id) not in self.poses[sequence]:
                target_id = frame_id

            rgb_path = os.path.join(
                self.data_root, "data_2d_raw", sequence, 'image_00/data_rect', target_id + ".png"
            )
            img = Image.open(rgb_path).convert("RGB")
            # Image augmentation
            if self.color_jitter is not None:
                img = self.color_jitter(img)
            # PIL to numpy
            img = np.array(img, dtype=np.float32, copy=False) / 255.0
            img = img[:self.img_H, :self.img_W, :]  # crop image

            image_list.append(self.normalize_rgb(img))

        image_tensor = torch.stack(image_list, dim=0) #[N, 3, 376, 1408]

        return image_tensor

    def get_gt_info(self, sequence, frame_id):
        """Get the ground truth.

        Args:
            sequence (str): sequence id,
            frame_id (str): frame id.

        Returns:
            array: target. 
        """
        # load full-range groundtruth
        target_1_path = os.path.join(self.label_root, sequence, frame_id + "_1_1.npy")
        target = np.load(target_1_path)
        # short-range groundtruth
        if self.eval_range == 25.6:
            target[128:, :, :] = 255
            target[:, :64, :] = 255
            target[:, 192:, :] = 255

        elif self.eval_range == 12.8:
            target[64:, :, :] = 255
            target[:, :96, :] = 255
            target[:, 160:, :] = 255

        return target

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_name='ssc',
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in SemanticKITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        if results is None:
            logger.info('Skip Evaluation')

        if 'ssc_scores' in results:
            # for single-GPU inference
            ssc_scores = results['ssc_scores']
            class_ssc_iou = ssc_scores['iou_ssc'].tolist()
            res_dic = {
                "SC_Precision": ssc_scores['precision'].item(),
                "SC_Recall": ssc_scores['recall'].item(),
                "SC_IoU": ssc_scores['iou'],
                "SSC_mIoU": ssc_scores['iou_ssc_mean'],
            }
        else:
            # for multi-GPU inference
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

        for name, iou in zip(self.class_names, class_ssc_iou):
            res_dic["SSC_{}_IoU".format(name)] = iou

        eval_results = {}
        for key, val in res_dic.items():
            eval_results['kitti360_{}'.format(key)] = round(val * 100, 2)

        eval_results['kitti360_combined_IoU'] = eval_results['kitti360_SC_IoU'] + eval_results['kitti360_SSC_mIoU']

        if logger is not None:
            logger.info('SSCBench-KITTI-360 SSC Evaluation')
            logger.info(eval_results)

        return eval_results

