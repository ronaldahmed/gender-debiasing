import json
import numpy as np
import os
name = os.path.join("semi_annotate","gender_words_dist_for_w2v.json")

def get_gendered_words ():
    f= open(name, "r"). read()
    _dct = json.loads(f)
    vals = []
    cnts = []
    for k, v in _dct.items():
        v, cnt = v
        cnts.append(cnt)

    cnts=sorted(cnts)
    ## Only get words in top 75% frequency
    thresh_cnt = cnts[len(cnts)//100*75]

    for k, v in _dct.items():
        v, cnt = v
        if cnt>thresh_cnt:
            vals.append(v)

    vals=sorted(vals)

    _mean = np.mean(vals)
    _std = np.std(vals)
    gwords_pos = []
    gwords_neg = []
    prate= 0.05
    nrate = 0.025
    lowerbound = vals[int(len(vals)*nrate)]
    upperbound = vals[-int(len(vals)*prate)]
    for k, v in _dct.items():
        v, cnt = v
        #print(cnt)
        if (v < lowerbound or v > upperbound) and cnt > thresh_cnt:  # np.abs(v - _mean) > 2 * _std
            if v - _mean>0:
                gwords_pos.append(k)
            else:
                gwords_neg.append(k)
    #return set(gwords_neg), set(gwords_pos)
    return gwords_neg, gwords_pos

if __name__=="__main__":
    name = "gender_words_dist_for_w2v.json"
    _n, _p = get_gendered_words()
    print("neg", list(_n)[:200], len(_n))
    print("pos", list(_p)[:200], len(_p))
