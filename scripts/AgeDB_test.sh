#!/bin/bash
#SBATCH --job-name=AgeDB_test    ## job name
#SBATCH --nodes=1                ## 索取 2 節點
#SBATCH --ntasks-per-node=1      ## 每個節點運行 8 srun tasks
#SBATCH --cpus-per-task=4        ## 每個 srun task 索取 4 CPUs
#SBATCH --gres=gpu:1             ## 每個節點索取 8 GPUs
#SBATCH --account="MST110260"   ## PROJECT_ID 請填入計畫ID(ex: MST108XXX)，扣款也會根據此計畫ID
#SBATCH --partition=gp1d        ## gtest 為測試用 queue，後續測試完可改 gp1d(最長跑1天)、gp2d(最長跑2天)、gp4d(最長跑4天)
#SBATCH --output=/work/jgtf0322/Rank-N-Contrast/logs/test.log

cd /tmp2/jeffwang/Rank-N-Contrast

# ckpt="/tmp2/jeffwang/Rank-N-Contrast/checkpoints/deltaorder/AgeDB_resnet18_ep_400_norm_l2_delta_0.1_trial_0/curr_last.pth"
# ckpt="/tmp2/jeffwang/Rank-N-Contrast/checkpoints/pointwise/norm_l2_obj_corr/last.pth"
# ckpt="/tmp2/jeffwang/Rank-N-Contrast/checkpoints/L1/best.pth"
# ckpt="/tmp2/jeffwang/Rank-N-Contrast/checkpoints/deltaorder/AgeDB_resnet18_ep_400_norm_l2_delta_0.1_trial_0/last.pth"
# ckpt="/tmp2/jeffwang/Rank-N-Contrast/checkpoints/pairwise/PwR_AgeDB_resnet18_ep_400_norm_l2_obj_correlation_trial_0/last.pth"
ckpt="/tmp2/jeffwang/Rank-N-Contrast/checkpoints/ProbRank/AgeDB_resnet18_ep_400_t_4_seed_3_trial_0/best_nbr_ydiff.pth"

python AgeDB_exp.py \
    --ckpt ${ckpt} \
    --add_regressor \
    --regerssor_ckpt "/tmp2/jeffwang/Rank-N-Contrast/checkpoints/pairwise/AgeDB_resnet18_ep_400_delta_1.0_obj_l1_seed_2_trial_0/Regressor_AgeDB_ep_100_lr_0.05_d_0.2_wd_0_mmt_0.9_bsz_64_bias_True_trial_0_best.pth" 
    # --sup_resnet
    # --umap_pic_name "pairwise/delta03_seed322"
    
    
    
    
