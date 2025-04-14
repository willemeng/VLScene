# VLScene
Official PyTorch Implementation of “VLScene: Vision-Language Guidance Distillation for Camera-based 3D Semantic Scene Completion”**(AAAI 2025 Oral)**, Paper is on [arxiv](https://arxiv.org/abs/2503.06219).


# Abstract
Camera-based 3D semantic scene completion (SSC) provides dense geometric and semantic perception for autonomous driving. However, images provide limited information making the model susceptible to geometric ambiguity caused by occlusion and perspective distortion. Existing methods often lack explicit semantic modeling between objects, limiting their perception of 3D semantic context. To address these challenges, we propose a novel method VLScene: Vision-Language Guidance Distillation for Camera-based 3D Semantic Scene Completion. The key insight is to use the vision-language model to introduce high-level semantic priors to provide the object spatial context required for 3D scene understanding. Specifically, we design a vision-language guidance distillation process to enhance image features, which can effectively capture semantic knowledge from the surrounding environment and improve spatial context reasoning. In addition, we introduce a geometric-semantic sparse awareness mechanism to propagate geometric structures in the neighborhood and enhance semantic information through contextual sparse interactions. Experimental results demonstrate that VLScene achieves rank-1st performance on challenging benchmarks—SemanticKITTI and SSCBench-KITTI-360, yielding remarkably mIoU scores of 17.52 and 19.10, respectively.

# Step-by-step Installation Instructions

Following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation

**a. Create a conda virtual environment and activate it.**
python > 3.7 may not be supported, because installing open3d-python with py>3.7 causes errors.
```shell
conda create -n vlscene python=3.7 -y
conda activate vlscene
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

**c. Install gcc>=5 in conda env (optional).**
I do not use this step.
```shell
conda install -c omgarcia gcc-6 # gcc-6.2
```

**c. Install mmcv-full.**
```shell
pip install mmcv-full==1.4.0
```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**e. Install mmdet3d from source code.**

Refer to occformer's mmdetection3d installation method.
Please check your CUDA version for [mmdet3d](https://github.com/open-mmlab/mmdetection3d/issues/2427) if encountered import problem. 

**f. Install other dependencies.**
```shell
pip install timm
pip install open3d-python
pip install PyMCubes
pip install spconv-cu113==2.3.6
pip install clip==1.0
```

# Prepare Data

- **a. You need to download**

     - The **Odometry calibration** (Download odometry data set (calibration files)) and the **RGB images** (Download odometry data set (color)) from [KITTI Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php), extract them to the folder `data/occupancy/semanticKITTI/RGB/`.
     - The **Velodyne point clouds** (Download [data_odometry_velodyne](http://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip)) and the **SemanticKITTI label data** (Download [data_odometry_labels](http://www.semantic-kitti.org/assets/data_odometry_labels.zip)) for sparse LIDAR supervision in training process, extract them to the folders ``` data/lidar/velodyne/ ``` and ``` data/lidar/lidarseg/ ```, separately. 


- **b. Prepare KITTI voxel label (see sh file for more details)**
```
bash process_kitti.sh
```

# Pretrained Model

Download the [RepViT-M2.3-300e](https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m2_3_distill_300e.pth), the [LSeg model](https://drive.google.com/file/d/1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb/view?usp=sharing) and [VLScene](https://pan.baidu.com/s/1Axv1TzH8Xi415a86Up5V9w) 提取码: k9vy, put them into `./checkpoints`.


# Training & Evaluation

## Single GPU
- **Train with single GPU:**
```
export PYTHONPATH="."  
python tools/train.py   \
            projects/configs/occupancy/semantickitti/vlscene.py
```

- **Evaluate with single GPUs:**
```
export PYTHONPATH="."  
python tools/test.py  \
            projects/configs/occupancy/semantickitti/vlscene.py \
            pretrain/checkpoint.pth 
```


## Multiple GPUS
- **Train with n GPUs:**
```
bash run.sh  \
        projects/configs/occupancy/semantickitti/vlscene.py n
```

- **Evaluate with n GPUs:**
```
 bash tools/dist_test.sh  \
            projects/configs/occupancy/semantickitti/vlscene.py \
            pretrain/checkpoint.pth  n
```

# Citation
If you find this project useful in your research, please consider cite:
```
@inproceedings{wang2025vlscene,
  title={VLScene: Vision-Language Guidance Distillation for Camera-Based 3D Semantic Scene Completion},
  author={Wang, Meng and Pi, Huilong and Li, Ruihui and Qin, Yunchuan and Tang, Zhuo and Li, Kenli},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={8},
  pages={7808--7816},
  year={2025}
}
```

# Acknowledgements
Many thanks to these excellent open source projects: 
- [MonoScene](https://github.com/astra-vision/MonoScene)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [StereoScene](https://github.com/Arlo0o/StereoScene/tree/main)
- [OccFormer](https://github.com/noticeable/OccFormer/tree/main)
- [RepViT](https://github.com/THU-MIG/RepViT)
- [KYN](https://github.com/ruili3/Know-Your-Neighbors) 
