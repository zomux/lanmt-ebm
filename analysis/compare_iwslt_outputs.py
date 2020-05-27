import numpy as np
import editdistance

paths = "rep_removed_delta_1 rep_removed_fakegrad_0 rep_removed_fakegrad_8".split()
files = ["/scratch/yl1363/lanmt-ebm/hyp/{}.txt".format(xx) for xx in paths]
files = ["/scratch/yl1363/corpora/iwslt/iwslt16_ende/test/test.de", "/scratch/yl1363/corpora/iwslt/iwslt16_ende/test/test.en"] + files
files = [open(ff, 'r').readlines() for ff in files]
all_files = [[line.strip() for line in ff if line.strip() != ""] for ff in files]
src, trg, mean, delta, sgd = all_files

lens = len(src)
assert ( len( set( [len(xx) for xx in [src, trg, mean, delta, sgd]] ) ) == 1 )

dic = {}

for ii in range(lens):
    #ed = editdistance.eval(mean[ii], sgd[ii])
    ed = editdistance.eval(trg[ii], sgd[ii])
    new = {
        "src": src[ii],
        "trg": trg[ii],
        "mean": mean[ii],
        "delta": delta[ii],
        "sgd": sgd[ii]
    }
    if ed in dic:
        dic[ed].append(new)
    else:
        dic[ed] = [new]
keys_sorted = sorted(list(dic.keys()))

#for ed in keys_sorted[::-1]:
for ed in keys_sorted:
    lst = dic[ed]
    for items in lst:
        src_ = items["src"]
        trg_ = items["trg"]
        mean_ = items["mean"]
        delta_ = items["delta"]
        sgd_ = items["sgd"]
        if mean_ == sgd_:
            continue

        print ("ED    :", ed)
        print ("SRC   :", src_)
        print ("TRG   :", trg_)
        print ("MEAN  :", mean_)
        print ("DELTA :", delta_)
        print ("SGD   :", sgd_)
        import ipdb; ipdb.set_trace()
