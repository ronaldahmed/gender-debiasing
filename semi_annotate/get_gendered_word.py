import json
import numpy as np
import os
name = os.path.join("semi_annotate","gender_words_dist_for_w2v.json")

def get_gendered_words ():
    f= open(name, "r"). read()
    _dct = json.loads(f)
    vals = []
    for k, v in _dct.items():
        vals.append(v)

    _mean = np.mean(vals)
    _std = np.std(vals)
    gwords_pos = []
    gwords_neg = []
    for k, v in _dct.items():
        if np.abs(v - _mean) > 2 * _std:
            if v - _mean>0:
                gwords_pos.append(k)
            else:
                gwords_neg.append(k)
    return set(gwords_neg), set(gwords_pos)

if __name__=="__main__":
    name = "gender_words_dist_for_w2v.json"
    _n, _p = get_gendered_words()
    print("neg", list(_n)[:200], len(_n))
    print("pos", list(_p)[:200], len(_p))
