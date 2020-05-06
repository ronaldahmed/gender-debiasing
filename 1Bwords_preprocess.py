# %% imports

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'osx')
import nltk
nltk.download('wordnet')
from tqdm import tqdm
from nltk.corpus import wordnet as wn
import yaml
import spacy
import lemminflect
from multiprocessing import Pool
import re

# %% FUNCTIONS

nlp = spacy.load("en_core_web_sm")
female_lookup_words = ["female", "woman",  "women",  "girl", "sister", "daughter", "mother",  "mom", "grandmother", "wife"]
male_lookup_words = ["male", "man", "men", "boy", "brother", "son", "father", "dad", "grandfather", "husband"]


def get_gender(vocab, verbose=False):
    # vocab should be a set of words that have at least one synset in wordnet
    fwords = {}
    mwords = {}
    nwords = {}

    for token in tqdm(vocab, total=len(vocab), desc='Getting gender from WN'):
        if len(token) == 1:
            # removing single consonant
            continue
        if any([len(re.findall(r"\b{}s?\b".format(pattern), token)) for pattern in female_lookup_words]):
            fwords[token] = wn.synsets(token)[0].definition()
        elif any([len(re.findall(r"\b{}s?\b".format(pattern), token)) for pattern in male_lookup_words]):
            mwords[token] = wn.synsets(token)[0].definition()
        else:
            definition = wn.synsets(token)[0].definition()  # the first synset is supposed to be most used definition
            male_freq = sum([len(re.findall(r"\b{}s?\b".format(pattern), definition)) for pattern in male_lookup_words])
            female_freq = sum([len(re.findall(r"\b{}s?\b".format(pattern), definition)) for pattern in female_lookup_words])
            if male_freq > female_freq:
                mwords[token] = definition
            elif male_freq == female_freq:
                nwords[token] = definition
            else:
                fwords[token] = definition
    if verbose:
        print(f"len of mwords: {len(mwords)} --- len of fwords: {len(fwords)} --- len of nwords: {len(nwords)}")

    with open("./data/female_words.yaml", "w") as ff:
        yaml.dump(fwords, ff)
    with open("./data/male_words.yaml", "w") as ff:
        yaml.dump(mwords, ff)
    with open("./data/neutral_words.yaml", "w") as ff:
        yaml.dump(nwords, ff)

    return fwords, mwords, nwords


def extract_words(filename, vocab, wn_vocab, mode='r'):
    assert type(vocab) is set, "vocab argument should be set"
    assert type(wn_vocab) is set, "wn_vocab argument should be set"

    with open(filename, mode) as f_in:
        fname = filename.split("/")[-1]
        for sentence in tqdm(f_in, f"File {fname}: "):
            sent = nlp(sentence)
            for token in sent:
                if token.is_stop or not token.is_alpha:
                    continue
                lemma = token._.lemma().lower()
                if lemma not in vocab:
                    vocab.add(lemma)
                else:
                    continue
                w_synsets = wn.synsets(lemma)
                if w_synsets and lemma not in wn_vocab:
                    wn_vocab.add(lemma)
    return vocab, wn_vocab


# %% main

FNAME_TRAINING = '/Users/kbello/Downloads/gender_data/training/news.en-000{}{}-of-00100'
FNAME_HELDOUT = '/Users/kbello/Downloads/gender_data/heldout/'
FNAME_PLAIN_WORDS = './data/lemmatized_words.txt'
FNAME_NOT_FOUND = './data/not_found_words.txt'
NCORES = 8
results = {}


def main():
    r_id = 0
    pool = Pool(NCORES)

    for i in tqdm(range(1, 48)):
        r_id += 1
        results[r_id] = pool.apply_async(
            extract_words,
            args=(FNAME_TRAINING.format(i//10, i%10), set(), set()),
        ) 

    r_id += 1
    results[r_id] = pool.apply_async(
        extract_words,
        args=(FNAME_HELDOUT + 'news.en-00000-of-00100', set(), set()),
    ) 
    
    for i in tqdm(range(50)):
        r_id += 1
        results[r_id] = pool.apply_async(
            extract_words,
            args=(FNAME_HELDOUT + 'news.en.heldout-000{}{}-of-00050'.format(i//10, i%10), set(), set()),
        )
    
    print(f"total # of files analyzed: {r_id}")

    pool.close()
    pool.join()

    vocab, wn_vocab = set(), set()
    for _, result in tqdm(results.items(), total=r_id):
        vocab_idx, wn_vocab_idx = result.get()
        vocab = vocab.union(vocab_idx)
        wn_vocab = wn_vocab.union(wn_vocab_idx)

    with open("./data/wn_vocab.yaml", "w") as ff:
        yaml.dump(wn_vocab, ff)
    
    with open("./data/all_vocab.yaml", "w") as ff:
        yaml.dump(vocab, ff)

    print(f"# of words in total: {len(vocab)}")
    print(f"# of words in wordnet: {len(wn_vocab)}")




# %%



# %%

if __name__ == "__main__":
    main()