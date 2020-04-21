# Real grad
python run2.py --opt_dtok iwslt16_deen --opt_batchtokens 4092 --opt_distill --opt_annealbudget --opt_tied --opt_scorenet --train --opt_decoder fixed --opt_noise rand --tensorboard --opt_cosine TC --opt_modeltype realgrad

# Fake grad
python run2.py --opt_dtok iwslt16_deen --opt_batchtokens 4092 --opt_distill --opt_annealbudget --opt_tied --opt_scorenet --train --opt_decoder fixed --opt_noise rand --tensorboard --opt_cosine TC --opt_modeltype fakegrad
