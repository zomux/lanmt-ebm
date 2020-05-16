#!/usr/bin/zsh

export CUDA_VISIBLE_DEVICES=0

cd /misc/vlgscratch4/ChoGroup/jason/lanmt-ebm/

# 05/15 prince. train EBM on WMT'16 Ro->En
python run_ebm.py --opt_dtok wmt16_roen --opt_batchtokens 8192 --opt_distill --opt_scorenet --train --tensorboard --opt_modeltype fakegrad --opt_train_delta_steps 4 --opt_fixbug2 --opt_direction_n_layers 4
python run_ebm.py --opt_dtok wmt16_roen --opt_batchtokens 8192 --opt_distill --opt_scorenet --train --tensorboard --opt_modeltype fakegrad --opt_train_delta_steps 4 --opt_fixbug2 --opt_direction_n_layers 5
python run_ebm.py --opt_dtok wmt16_roen --opt_batchtokens 8192 --opt_distill --opt_scorenet --train --tensorboard --opt_modeltype fakegrad --opt_train_delta_steps 4 --opt_fixbug2 --opt_direction_n_layers 6
#python run_ebm.py --opt_dtok wmt16_roen --opt_batchtokens 8192 --opt_distill --opt_scorenet --opt_modeltype fakegrad --opt_train_delta_steps 4 --opt_fixbug2 --opt_direction_n_layers 6 --batch_test --evaluate

# 05/13 prince. train LVM on WMT'16 Ro->En
python run_lvm.py --opt_dtok wmt16_roen --opt_batchtokens 4092 --opt_distill --opt_latentdim 256 --opt_priorl 6 --opt_decoderl 6 --opt_hiddensz 256 --opt_embedsz 256 --opt_heads 8 --opt_fixbug2 --opt_lr 0.0003 --train --tensorboard
python run_lvm.py --opt_dtok wmt16_roen --opt_batchtokens 4092 --opt_distill --opt_latentdim 8 --opt_priorl 6 --opt_decoderl 6 --opt_hiddensz 256 --opt_embedsz 256 --opt_heads 8 --opt_fixbug2 --opt_lr 0.0003 --train --tensorboard

# 05/12 EBM (energy model, not score) trained on IWSLT, with both ConvNet and Linear parameterization, new loss function (not cosine sim)
python run_ebm.py --opt_dtok iwslt16_deen --opt_latentdim 8 --opt_batchtokens 4092 --opt_distill --opt_scorenet --train --tensorboard --opt_train_delta_steps 4 --opt_modeltype realgrad --opt_losstype original --opt_ebm_lr 0.0003 --opt_ebm_useconv --opt_direction_n_layers 4
python run_ebm.py --opt_dtok iwslt16_deen --opt_latentdim 8 --opt_batchtokens 4092 --opt_distill --opt_scorenet --train --tensorboard --opt_train_delta_steps 4 --opt_modeltype realgrad --opt_losstype original --opt_ebm_lr 0.0003 --opt_direction_n_layers 4

# 05/10 prince without length normalization
python run_ebm.py --opt_dtok iwslt16_deen --opt_latentdim 2 --opt_batchtokens 4092 --opt_distill --opt_scorenet --train --tensorboard --opt_train_delta_steps 4 --opt_modeltype fakegrad --opt_losstype original --opt_fixbug2 --opt_ebm_lr 0.0003
python run_ebm.py --opt_dtok iwslt16_deen --opt_latentdim 2 --opt_batchtokens 4092 --opt_distill --opt_scorenet --train --tensorboard --opt_train_delta_steps 4 --opt_modeltype realgrad --opt_losstype original --opt_fixbug2 --opt_ebm_lr 0.0003

# 05/09 cassio with length normalization
python run_ebm.py --opt_dtok iwslt16_deen --opt_latentdim 2 --opt_batchtokens 4092 --opt_distill --opt_scorenet --train --tensorboard --opt_train_delta_steps 4 --opt_modeltype fakegrad --opt_losstype original --opt_fixbug2 --opt_ebm_lr 0.0003
python run_ebm.py --opt_dtok iwslt16_deen --opt_latentdim 2 --opt_batchtokens 4092 --opt_distill --opt_scorenet --train --tensorboard --opt_train_delta_steps 4 --opt_modeltype realgrad --opt_losstype original --opt_fixbug2 --opt_ebm_lr 0.0003

# 05/08
python run_ebm.py --opt_dtok iwslt16_deen --opt_latentdim 2 --opt_batchtokens 4092 --opt_distill --opt_scorenet --train --tensorboard --opt_train_delta_steps 4 --opt_modeltype fakegrad --opt_fixbug2 --opt_ebm_lr 0.0003 --opt_train_sgd_steps 1 --opt_train_step_size 0.8
python run_ebm.py --opt_dtok iwslt16_deen --opt_latentdim 2 --opt_batchtokens 4092 --opt_distill --opt_scorenet --train --tensorboard --opt_train_delta_steps 4 --opt_modeltype fakegrad --opt_fixbug2 --opt_ebm_lr 0.0003
python run_ebm.py --opt_dtok iwslt16_deen --opt_latentdim 2 --opt_batchtokens 4092 --opt_distill --opt_scorenet --train --tensorboard --opt_train_delta_steps 4 --opt_modeltype fakegrad --opt_ebm_lr 0.0003

# 05/07
python run_lvm.py --opt_dtok iwslt16_deen --opt_batchtokens 4092 --opt_distill --opt_annealbudget --opt_fastanneal --opt_latentdim 2 --opt_fixbug2 --opt_lr 0.0003 --train --tensorboard
python run_lvm.py --opt_dtok iwslt16_deen --opt_batchtokens 4092 --opt_distill --opt_annealbudget --opt_latentdim 2 --opt_fixbug2 --opt_lr 0.0003 --train --tensorboard
python run_lvm.py --opt_dtok iwslt16_deen --opt_batchtokens 4092 --opt_distill --opt_annealbudget --opt_latentdim 2 --opt_fixbug2 --opt_lr 0.0010 --train --tensorboard
python run_lvm.py --opt_dtok iwslt16_deen --opt_batchtokens 4092 --opt_distill --opt_annealbudget --opt_fastanneal --opt_latentdim 2 --opt_fixbug2 --opt_lr 0.0010 --train --tensorboard

# 05/06
python run2.py --opt_dtok iwslt16_deen --opt_batchtokens 4092 --opt_distill --opt_annealbudget --opt_tied --opt_scorenet --train --tensorboard --opt_train_delta_steps 4 --opt_modeltype fakegrad --opt_ebm_lr 0.0003 --opt_train_sgd_steps 1 --opt_train_step_size 0.8
python run2.py --opt_dtok iwslt16_deen --opt_batchtokens 4092 --opt_distill --opt_annealbudget --opt_tied --opt_scorenet --train --tensorboard --opt_train_delta_steps 4 --opt_modeltype realgrad --opt_ebm_lr 0.0003 --opt_train_sgd_steps 1 --opt_train_step_size 0.8


#python run2.py --opt_dtok iwslt16_deen --opt_batchtokens 4092 --opt_distill --opt_annealbudget --opt_tied --opt_scorenet --train --tensorboard --opt_train_delta_steps 4 --opt_modeltype realgrad --opt_ebm_lr 0.0003 
#python run2.py --opt_dtok iwslt16_deen --opt_batchtokens 4092 --opt_distill --opt_annealbudget --opt_tied --opt_scorenet --train --tensorboard --opt_train_delta_steps 4 --opt_modeltype fakegrad --opt_ebm_lr 0.0003
#python run3.py --opt_dtok iwslt16_deen --opt_batchtokens 4092 --opt_distill --opt_annealbudget --opt_tied --opt_scorenet --train --tensorboard --opt_train_delta_steps 4 --opt_modeltype fakegrad --opt_ebm_lr 0.0003 --opt_direction_n_layers 6 --opt_magnitude_n_layers 2
#python run2.py --opt_dtok iwslt16_deen --opt_batchtokens 4092 --opt_distill --opt_annealbudget --opt_tied --opt_scorenet --train --tensorboard --opt_train_delta_steps 4 --opt_modeltype fakegrad --opt_ebm_lr 0.0003 --opt_train_sgd_steps 1 --opt_train_step_size 0.8
#python run2.py --opt_dtok iwslt16_deen --opt_batchtokens 4092 --opt_distill --opt_annealbudget --opt_tied --opt_scorenet --train --tensorboard --opt_train_delta_steps 4 --opt_modeltype fakegrad --opt_ebm_lr 0.0003 --opt_train_interpolate_ratio 0.5
#python run2.py --opt_dtok iwslt16_deen --opt_batchtokens 4092 --opt_distill --opt_annealbudget --opt_tied --opt_scorenet --train --tensorboard --opt_train_delta_steps 4 --opt_modeltype fakegrad --opt_ebm_lr 0.0003 --opt_direction_n_layers 1 --opt_magnitude_n_layers 1
#python run2.py --opt_dtok iwslt16_deen --opt_batchtokens 4092 --opt_distill --opt_annealbudget --opt_tied --opt_scorenet --train --tensorboard --opt_train_delta_steps 4 --opt_modeltype fakegrad --opt_ebm_lr 0.0003 --opt_direction_n_layers 2 --opt_magnitude_n_layers 2
#python run2.py --opt_dtok iwslt16_deen --opt_batchtokens 4092 --opt_distill --opt_annealbudget --opt_tied --opt_scorenet --train --tensorboard --opt_train_delta_steps 4 --opt_modeltype fakegrad --opt_ebm_lr 0.0003 --opt_direction_n_layers 3 --opt_magnitude_n_layers 3
#python run2.py --opt_dtok iwslt16_deen --opt_batchtokens 4092 --opt_distill --opt_annealbudget --opt_tied --opt_scorenet --train --tensorboard --opt_train_delta_steps 4 --opt_modeltype fakegrad --opt_ebm_lr 0.0003 --opt_direction_n_layers 4 --opt_magnitude_n_layers 2

