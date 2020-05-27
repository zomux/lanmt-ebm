import numpy as np
import editdistance

paths = "hyp_delta_0 hyp_delta_1 hyp_delta_2 hyp_delta_4 hyp_delta_4 hyp_fakegrad_0 hyp_fakegrad_1 hyp_fakegrad_2 hyp_fakegrad_4 hyp_fakegrad_8".split()
files = ["/scratch/yl1363/lanmt-ebm/hyp/{}.txt".format(xx) for xx in paths]
files = ["/scratch/yl1363/corpora/iwslt/iwslt16_ende/test/test.de", "/scratch/yl1363/corpora/iwslt/iwslt16_ende/test/test.en"] + files
files = [open(ff, 'r').readlines() for ff in files]
all_files = [[line.strip() for line in ff if line.strip() != ""] for ff in files]
src, trg, d0, d1, d2, d4, d8, s0, s1, s2, s4, s8 = all_files

def get_num_rep(lines):
    cnt = 0
    for line_idx, line in enumerate(lines):
        words = line.strip().split()
        for idx in range(len(words)-1):
            if words[idx] == words[idx+1]:
                cnt += 1
                #print (line, line_idx)
                break
    return cnt
print ([get_num_rep(lines) for lines in [src, trg]])
print ([get_num_rep(lines) for lines in [d0, d1, d2, d4, d8]])
print ([get_num_rep(lines) for lines in [s0, s1, s2, s4, s8]])

mean_d01 = np.array([editdistance.eval(x1, x2) for x1, x2 in zip(d0, d1)])
mean_d02 = np.array([editdistance.eval(x1, x2) for x1, x2 in zip(d0, d2)])
mean_d04 = np.array([editdistance.eval(x1, x2) for x1, x2 in zip(d0, d4)])
mean_d08 = np.array([editdistance.eval(x1, x2) for x1, x2 in zip(d0, d8)])

mean_s01 = np.array([editdistance.eval(x1, x2) for x1, x2 in zip(d0, s1)])
mean_s02 = np.array([editdistance.eval(x1, x2) for x1, x2 in zip(d0, s2)])
mean_s04 = np.array([editdistance.eval(x1, x2) for x1, x2 in zip(d0, s4)])
mean_s08 = np.array([editdistance.eval(x1, x2) for x1, x2 in zip(d0, s8)])

print (mean_d01.mean(), mean_d02.mean(), mean_d04.mean(), mean_d08.mean())
print (mean_s01.mean(), mean_s02.mean(), mean_s04.mean(), mean_s08.mean())

ref_d01 = np.array([editdistance.eval(x1, x2) for x1, x2 in zip(trg, d1)])
ref_d02 = np.array([editdistance.eval(x1, x2) for x1, x2 in zip(trg, d2)])
ref_d04 = np.array([editdistance.eval(x1, x2) for x1, x2 in zip(trg, d4)])
ref_d08 = np.array([editdistance.eval(x1, x2) for x1, x2 in zip(trg, d8)])

ref_s01 = np.array([editdistance.eval(x1, x2) for x1, x2 in zip(trg, s1)])
ref_s02 = np.array([editdistance.eval(x1, x2) for x1, x2 in zip(trg, s2)])
ref_s04 = np.array([editdistance.eval(x1, x2) for x1, x2 in zip(trg, s4)])
ref_s08 = np.array([editdistance.eval(x1, x2) for x1, x2 in zip(trg, s8)])

print (ref_d01.mean(), ref_d02.mean(), ref_d04.mean(), ref_d08.mean())
print (ref_s01.mean(), ref_s02.mean(), ref_s04.mean(), ref_s08.mean())


import ipdb; ipdb.set_trace()
total_num = len(s0)
sgd_diff_idx = [idx for idx in range(len(step0)) if step0[idx] != sgd[idx]]
delta_diff_idx = [idx for idx in range(len(step0)) if step0[idx] != step1[idx]]
sgd_delta_same_idx = [idx for idx in range(len(step0)) if sgd[idx] == step1[idx]]
sgd_delta_same_but_diff_idx = [idx for idx in range(len(step0)) if sgd[idx] == step1[idx] and sgd[idx] != step0[idx]]

print (len(sgd_diff_idx))
print (len(delta_diff_idx))
print (len(sgd_delta_same_idx))
print (len(sgd_delta_same_but_diff_idx))
exit()
import ipdb; ipdb.set_trace()

for ii in sgd_diff_idx:
  print ("SRC   :", src[ii])
  print ("TRG   :", trg[ii])
  print ("STEP0 :", step0[ii])
  print ("STEP1 :", step1[ii])
  print ("SGD   :", sgd[ii])
  import ipdb; ipdb.set_trace()
