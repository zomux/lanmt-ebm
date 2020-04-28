#!/usr/bin/zsh

export CUDA_VISIBLE_DEVICES=0

cd /misc/vlgscratch4/ChoGroup/jason/lanmt-ebm/

#python run2.py --opt_dtok iwslt16_deen --opt_batchtokens 4092 --opt_distill --opt_annealbudget --opt_tied --opt_scorenet --train --tensorboard --opt_train_delta_steps 4 --opt_modeltype realgrad --opt_ebm_lr 0.0003 
#python run2.py --opt_dtok iwslt16_deen --opt_batchtokens 4092 --opt_distill --opt_annealbudget --opt_tied --opt_scorenet --train --tensorboard --opt_train_delta_steps 4 --opt_modeltype fakegrad --opt_ebm_lr 0.0003
python run2.py --opt_dtok iwslt16_deen --opt_batchtokens 4092 --opt_distill --opt_annealbudget --opt_tied --opt_scorenet --train --tensorboard --opt_train_delta_steps 4 --opt_modeltype fakegrad --opt_ebm_lr 0.0003 --opt_train_interpolate_ratio 0.5
