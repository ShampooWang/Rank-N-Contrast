#!/bin/bash
#SBATCH --job-name=AgeDB_test    ## job name
#SBATCH --nodes=1                ## 索取 2 節點
#SBATCH --ntasks-per-node=1      ## 每個節點運行 8 srun tasks
#SBATCH --cpus-per-task=4        ## 每個 srun task 索取 4 CPUs
#SBATCH --gres=gpu:1             ## 每個節點索取 8 GPUs
#SBATCH --account="MST110260"   ## PROJECT_ID 請填入計畫ID(ex: MST108XXX)，扣款也會根據此計畫ID
#SBATCH --partition=gp1d        ## gtest 為測試用 queue，後續測試完可改 gp1d(最長跑1天)、gp2d(最長跑2天)、gp4d(最長跑4天)
#SBATCH --output=/work/jgtf0322/Rank-N-Contrast/logs/test.log

cd /work/jgtf0322/Rank-N-Contrast
ckpt="/work/jgtf0322/Rank-N-Contrast/AgeDB_models/RnC_AgeDB_resnet18_ep_400_lr_0.5_d_0.1_wd_0.0001_mmt_0.9_bsz_256_aug_crop,flip,color,grayscale_temp_2_label_l1_feature_l2_trial_0/last.pth"


python AgeDB_test.py --ckpt ${ckpt}