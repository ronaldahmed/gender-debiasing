import json
import numpy as np
import os
from ordered_set import OrderedSet

def get_gendered_words(basedir = "semi_annotate"):
    neg = None
    pos = None
    for idx, name in enumerate(["w2v", "glove_twitter"]):  #
        name = os.path.join(basedir, "gender_words_dist_for_%s.json" % name)
        ## Mutate all word embeddings except for first one
        _neg, _pos = get_gendered_words_from_path(
            name, add_mutate_flag=(idx != 1))
        _neg = OrderedSet(_neg)
        _pos = OrderedSet(_pos)
        if neg is None:
            neg = _neg
            pos = _pos
        else:
            neg = neg.intersection(_neg)
            pos = pos.intersection(_pos)
    return neg, pos

def add_ids( _lst, word):
    _lst.append(word)

def add_mutate( _lst, word):
    added = False
    words = [word, word.replace("_", " ")]
    for word in words:

        added = added or add_ids(_lst, word)
        added = added or add_ids(_lst, word.lower())
        added = added or add_ids(_lst, word.upper())
        added = added or add_ids(
            _lst, word[0].upper()+word[1:].lower())
    if not added:
        #print(word)
        pass



def get_gendered_words_from_path (path, add_mutate_flag =True):
    f= open(path, "r"). read()
    _dct = json.loads(f)
    vals = []
    cnts = []
    for k, v in _dct.items():
        v, cnt = v
        cnts.append(cnt)

    cnts=sorted(cnts)
    ## Only get words in top x% frequency
    thresh_cnt = cnts[len(cnts)//100*25]

    for k, v in _dct.items():
        v, cnt = v
        if cnt>thresh_cnt:
            vals.append(v)

    vals=sorted(vals)

    _mean = np.mean(vals)
    _std = np.std(vals)
    gwords_pos = []
    gwords_neg = []
    prate= 0.1
    nrate = 0.1
    lowerbound = vals[int(len(vals)*nrate)]
    upperbound = vals[-int(len(vals)*prate)]
    for k, v in _dct.items():
        v, cnt = v
        #print(cnt)
        if (v < lowerbound or v > upperbound) and cnt > thresh_cnt:  # np.abs(v - _mean) > 2 * _std
            if v - _mean>0:
                #gwords_pos.append(k)
                if add_mutate_flag:
                    add_mutate(gwords_pos, k)
                else:
                    gwords_pos.append(k)
            else:
                #
                if add_mutate_flag:
                    add_mutate(gwords_neg, k)
                else:
                    gwords_neg.append(k)
    #return set(gwords_neg), set(gwords_pos)
    return gwords_neg, gwords_pos



if __name__=="__main__":
    name = "gender_words_dist_for_w2v.json"
    _n, _p = get_gendered_words(basedir = ".")
    print("neg", list(_n)[:200], len(_n))
    print("pos", list(_p)[:200], len(_p))
