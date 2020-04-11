# Using log p(y^ | z, x) as the target, without imitation learning
python run2.py --opt_dtok iwslt16_deen --opt_batchtokens 4092 --opt_distill --opt_annealbudget --opt_tied --opt_scorenet --train --opt_decoder fixed --opt_noise rand --opt_targets xent --opt_line_search_c 0.05 --tensorboard

# Using log p(y^ | z, x) as the target, with imitation learning
python run2.py --opt_dtok iwslt16_deen --opt_batchtokens 4092 --opt_distill --opt_annealbudget --opt_tied --opt_scorenet --train --opt_decoder fixed --opt_noise rand --opt_targets xent --opt_line_search_c 0.05 --opt_imitation --opt_imit_rand_steps 5 --tensorboard
