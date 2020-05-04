import gensim.downloader as api
from sklearn import decomposition
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

models = {"w2v": "word2vec-google-news-300",
            "glove_wiki": "glove-wiki-gigaword-300",
            "glove_twitter": "glove-twitter-200"}

    
def remove(string, subs):
    for sub in subs:
        string = string.replace(sub, "")
    return string
            

def get_model_vecs(model, words):
    #global words
    #_words = remove(words, [" ","\r","\n"]).split(",")
    assert model_name in ["w2v", "glove_wiki", "glove_twitter"]
    vecs = []
    for word in words:
        #if word in model.wv:
        vec = model.wv[word]
        #print(word,vec)    
        vecs.append(vec)
    return vecs

def normalized(vec):
    vec = vec / np.linalg.norm(vec)
    return vec


    
if __name__=="__main__":

    words = """he, his, her, she, him, man, women, men, woman, spokesman, wife, himself, son, mother, father, chairman,
        daughter, husband, guy, girls, girl, boy, boys, brother, spokeswoman, female, sister, male, herself, brothers, dad,
        actress, mom, sons, girlfriend, daughters, lady, boyfriend, sisters, mothers, king, businessman, grandmother,
        grandfather, deer, ladies, uncle, males, congressman, grandson, bull, queen, businessmen, wives, widow,
        nephew, bride, females, aunt, prostate cancer, lesbian, chairwoman, fathers, moms, maiden, granddaughter,
        younger brother, lads, lion, gentleman, fraternity, bachelor, niece, bulls, husbands, prince, colt, salesman, hers,
        dude, beard, filly, princess, lesbians, councilman, actresses, gentlemen, stepfather, monks, ex girlfriend, lad,
        sperm, testosterone, nephews, maid, daddy, mare, fiance, fiancee, kings, dads, waitress, maternal, heroine,
        nieces, girlfriends, sir, stud, mistress, lions, estranged wife, womb, grandma, maternity, estrogen, ex boyfriend,
        widows, gelding, diva, teenage girls, nuns, czar, ovarian cancer, countrymen, teenage girl, penis, bloke, nun,
        brides, housewife, spokesmen, suitors, menopause, monastery, motherhood, brethren, stepmother, prostate,
        hostess, twin brother, schoolboy, brotherhood, fillies, stepson, congresswoman, uncles, witch, monk, viagra,
        paternity, suitor, sorority, macho, businesswoman, eldest son, gal, statesman, schoolgirl, fathered, goddess,
        hubby, stepdaughter, blokes, dudes, strongman, uterus, grandsons, studs, mama, godfather, hens, hen, mommy,
        estranged husband, elder brother, boyhood, baritone, grandmothers, grandpa, boyfriends, feminism, countryman,
        stallion, heiress, queens, witches, aunts, semen, fella, granddaughters, chap, widower, salesmen, convent,
        vagina, beau, beards, handyman, twin sister, maids, gals, housewives, horsemen, obstetrics, fatherhood,
        councilwoman, princes, matriarch, colts, ma, fraternities, pa, fellas, councilmen, dowry, barbershop, fraternal,
        ballerina"""

    pairs = [["she", "he"], ["her", "his"], ["woman", "man"], ["Mary", "John"], ["herself", "himself"], ["daughter", "son"], ["mother", "father"], ["gal", "guy"], ["girl", "boy"], ["female", "male"], ["wife", "husband"], ["girlfriend", "boyfriend"] ,["sister", "brother"]]

    #os.chdir(os.path.join(".", "semi_annotate"))
    for model_name in ["w2v", "glove_wiki", "glove_twitter"]:
        model = api.load(models[model_name])
        print("model name: ", model_name)
        vecs = []
        for pair in pairs:
            # refer to code https://github.com/tolga-b/debiaswe/blob/master/debiaswe/we.py from paper man is to ...
            vec1, vec2 = get_model_vecs(model, pair)
            vec1 = normalized(vec1)
            vec2 = normalized(vec2)
            center = (vec1 + vec2) / 2
            vecs.append(vec1 - center)
            vecs.append(vec2 - center)
        pca = decomposition.PCA()
        pca.fit(vecs)

        #w2c = []
        #for item in model.wv.vocab:
        #    w2c.append((item, model.wv.vocab[item].count))
        
        print(pca.explained_variance_ratio_)
        mcp = pca.components_[0]

        #w2c.sort(reverse=True, key= lambda x: x[1] )
        projs = []
        proj_dict = {}
        for item in model.wv.vocab:
            proj = np.dot(mcp, normalized(model.wv[item]))
            projs.append(proj)
            proj_dict[item] = (proj, model.vocab[item].count)
        plt.figure(figsize=(10, 10))
        sns.distplot(projs, hist=True, kde=False, norm_hist=True,
                    color='blue',
                    hist_kws={'edgecolor': 'black'})
        plt.title('Histogram ')
        plt.xlabel('Projection')
        plt.ylabel('Density')
        plt.savefig("gender_words_dist_for_%s.png"%model_name)
        js = json.dumps(proj_dict)
        open("gender_words_dist_for_%s.json"%model_name, "w").write(js+"\r\n")

# dataset corpus in gensim
# wiki-english-20171001
