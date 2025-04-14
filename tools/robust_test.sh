#!/usr/bin/env bash

set -e
exeFunc(){
    corr=$1 
    severity_level=$2
    CUDA_VISIBLE_DEVICES=2  python tools/test.py ./projects/configs/occupancy/semantickitti/clipscene_fusion.py ./work_dirs/clip_kdscene_base_sparse/best_semkitti_SSC_mIoU_epoch_23_1666_4415.pth --corr $corr --severity_level $severity_level
}
seqs=("brightness" "contrast" "dark" "fog" "frost" "snow")

# seqs=("contrast" "dark" "fog" "frost "snow")
for i in ${seqs[@]}
do
    exeFunc $i  1 
    exeFunc $i  3 
    exeFunc $i  5 
done