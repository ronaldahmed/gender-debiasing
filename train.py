# -*- coding: utf-8 -*-

import os
import pickle
import random
import argparse
import torch
import numpy as np
import json

from tqdm import tqdm
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from model import Word2Vec, SGNS, GenderClassifier, GetGenderLoss


import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, default='sgns', help="experiment ID")
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--save_dir', type=str, default='./logs/', help="model directory path")

    parser.add_argument('--mode', type=str, default='train', help="model directory path [train,]")


    parser.add_argument('--e_dim', type=int, default=300, help="embedding dimension")
    parser.add_argument('--n_negs', type=int, default=20, help="number of negative samples")
    parser.add_argument('--epoch', type=int, default=101, help="number of epochs")
    parser.add_argument('--mb', type=int, default=4096, help="mini-batch size")
    parser.add_argument('--ss_t', type=float, default=1e-5, help="subsample threshold")
    parser.add_argument('--load_model', type=int, default=-1, help="model ID to load. use to train from pre-trained or for evaluation")

    parser.add_argument('--conti', action='store_true', help="continue learning")
    parser.add_argument('--weights', action='store_true', help="use weights for negative sampling")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")

    parser.add_argument('--DLossBeta', type=float, default=1., help="weight of Discriminator Loss in W2V training")
    parser.add_argument('--gc_dim', type=int, default=4, help="hidden layer units of gender classifier")
    parser.add_argument('--normal', action="store_true", help="training word embeddgings normally without adversarial part")
    return parser.parse_args()

class PermutedSubsampledCorpus(Dataset):

    def __init__(self, datapath, ws=None):
        data = pickle.load(open(datapath, 'rb'))
        if ws is not None:
            self.data = []
            for iword, owords in data:
                if random.random() > ws[iword]:
                    self.data.append((iword, owords))
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iword, owords = self.data[idx]
        return iword, np.array(owords, dtype=np.int64)

def get_pos_neg_iter(neg_ids, pos_ids, batch_size = 1024, repeat= True):
    neg_ids = np.array(list(neg_ids))
    pos_ids = np.array(list(pos_ids))
    while True:
        np.random.shuffle(neg_ids)
        np.random.shuffle(pos_ids)
        l = min(len(neg_ids), len(pos_ids))
        ## for each class, only half of samples are needed
        batch_sz = batch_size // 2
        batch = l  // batch_sz 

        for i in range(batch):
            # return id , label
            yield np.concatenate([neg_ids[i*batch_sz:(i+1)*batch_sz],pos_ids[i*batch_sz:(i+1)*batch_sz]], axis=0), np.array([0]*batch_sz+[1]*batch_sz)
        if not repeat:
            break

class common_prepare():
    def __init__(self, args):
        idx2word = pickle.load(
            open(os.path.join(args.data_dir, 'idx2word.dat'), 'rb'))
        word2idx = pickle.load(
            open(os.path.join(args.data_dir, 'word2idx.dat'), 'rb'))
        wc = pickle.load(open(os.path.join(args.data_dir, 'wc.dat'), 'rb'))
        wf = np.array([wc[word] for word in idx2word])
        wf = wf / wf.sum()
        ws = 1 - np.sqrt(args.ss_t / wf)
        ws = np.clip(ws, 0, 1)
        vocab_size = len(idx2word)
        weights = wf if args.weights else None
        if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)
        log_dir = os.path.join(args.save_dir, args.exp_id)
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        model = Word2Vec(vocab_size=vocab_size, embedding_size=args.e_dim)
        sgns = SGNS(embedding=model, vocab_size=vocab_size,
                    n_negs=args.n_negs, weights=weights)

        only_one_classifier = False
        ## create gender classifier, currently it is 2-layer mlp
        gc_i = GenderClassifier(args.e_dim, hidden_units=args.gc_dim)
        if only_one_classifier:
            gc_o = gc_i
        else:
            gc_o = GenderClassifier(args.e_dim, hidden_units=args.gc_dim)

        if args.load_model != -1:
            model_name = os.path.join(log_dir, "model_%d.pt" % args.load_model)
            sgns.load_state_dict(torch.load(model_name))

        if args.cuda:
            sgns = sgns.cuda()
            gc_i = gc_i.cuda()
            gc_o = gc_o.cuda()

        gc_loss = GetGenderLoss(embedding=model, iclassifier=gc_i,
                                oclassifier=gc_o, word2idx=word2idx)
        gc_iter = get_pos_neg_iter(
            gc_loss.neg_ids_train, gc_loss.pos_ids_train, batch_size=128)
        test_ids, test_labels = gc_loss.test_ids, torch.LongTensor(
            gc_loss.test_label).cuda()

        criterion = torch.nn.CrossEntropyLoss()
        optim = Adam(sgns.parameters())
        optim_D = Adam(list(gc_i.parameters()) +
                    list(gc_o.parameters()))  # , lr=1e-2)
        beta = args.DLossBeta

        localvars = {"idx2word": idx2word, "word2idx": word2idx, "model" : model, "sgns":sgns, "gc_i":gc_i, "gc_o":gc_o, "gc_loss":gc_loss, "gc_iter":gc_iter, 
                     "test_ids": test_ids, "test_labels": test_labels, "optim": optim, "criterion": criterion, "optim_D": optim_D, "beta": beta, "log_dir": log_dir}
        self.lst = []        
        for k,v in localvars.items():
            setattr(self, k, v)
            self.lst.append(v)



def train(args):
    d = common_prepare(args)
    idx2word, word2idx, model, sgns, gc_i, gc_o, gc_loss, gc_iter, test_ids, test_labels, optim, criterion, optim_D, beta, log_dir = d.lst
        
    logf = open("EXP_%s.log" % args.exp_id, "w")
    if args.load_model != -1:
        optimpath = os.path.join(log_dir, "optim_%d.pt" % args.load_model)
        optim.load_state_dict(torch.load(optimpath))

    for epoch in range(args.epoch):
        dataset = PermutedSubsampledCorpus(os.path.join(args.data_dir, 'train.dat'))
        dataloader = DataLoader(dataset, batch_size=args.mb, shuffle=True)
        total_batches = int(np.ceil(len(dataset) / args.mb))
        pbar = tqdm(dataloader)
        pbar.set_description("[Epoch {}]".format(epoch))
        cnt = 0
        for iword, owords in pbar:
            
            ## The loss for embedding to update, and the embeddding is to fool the gender classifier to make wrong classification
            optim.zero_grad()
            Eloss1 = sgns(iword, owords)
            # They have batch normalization during training
            gc_i.eval()
            gc_o.eval()

            gci_acc = (torch.argmax(gc_i(model.forward_i(test_ids)), dim=-1) == test_labels).float().mean()

            Eloss = Eloss1
            Eloss2 = beta * gc_loss.forward_E(iword, owords[:, 4:5])


            if epoch>=0 and not args.normal:
                # for normal training this part is ignored
                # warmup for classification
                # (- gc_loss.forward_D(iword, owords[:, 4:5]))  #
                Eloss += Eloss2

            Eloss.backward()
            optim.step()
            #pbar.set_postfix(loss=loss.item())

            """"""
            ## The loss for gender classifier
            if epoch>=0:
                #warmup for w2v
                gc_i.train()
                gc_o.train()
                """
                BatchSize = 512
                for steps in range(4096//BatchSize):
                    start = BatchSize*steps
                    end = BatchSize*(steps+1)
                    optim_D.zero_grad()
                    Dloss, acci, acco = gc_loss.forward_D(
                        iword[start:end], owords[start:end,4:5], acc=True)
                    Dloss.backward()
                    optim_D.step()
                """
                ids, labels = next(gc_iter)
                labels = torch.LongTensor(labels).cuda()
                Dloss1 = criterion(gc_i(model.forward_i(ids)), labels)
                Dloss2 = criterion(gc_o(model.forward_o(ids)), labels)
                Dloss = Dloss1 + Dloss2
                optim_D.zero_grad()
                Dloss.backward()
                optim_D.step()

                rd = dict(  # acci=acci.item(),
                    #acco=acco.item(),
                    acci=gci_acc.item(),
                    ## gendered words percentage or signal rate
                    sr=float(gc_loss.effect_words) / \
                    (gc_loss.tot_words + 1e-5),
                    ## pos gendered words percentage or positve rate
                    pr=float(gc_loss.pos_words) / \
                    (gc_loss.effect_words+1e-5),
                    Dloss=Dloss.item(),  # (Dloss1.item(), Dloss2.item()),
                    Eloss1=Eloss1.item(),
                    Eloss2=Eloss2.item())

                pbar.set_postfix(rd)

            #if cnt > 2:
            #    break
            cnt += 1
        rd["epoch"] = epoch
        logf.write(json.dumps(rd)+"\r\n")
        logf.flush()

        ## run eval here & save progress
        if epoch % 10==0:
            model_name = os.path.join(log_dir, "model_%d.pt" % epoch)
            optim_name = os.path.join(log_dir, "optim_%d.pt" % epoch)
            emb_name = os.path.join(log_dir, "emb_%d.vec" % epoch)
            gc_i_name = os.path.join(log_dir, "gci_%d.pt" % epoch)

            #print("-> sanity check")
            #pdb.set_trace()

            torch.save(sgns.state_dict(),model_name)
            torch.save(optim.state_dict(),optim_name)
            torch.save(gc_i.state_dict(),gc_i_name)
            dump_embeddings_txt(model.ivectors.weight.data.cpu().numpy(),
                        emb_name,idx2word)

    # pickle.dump(idx2vec, open(os.path.join(args.data_dir, 'idx2vec.dat'), 'wb'))
    # torch.save(sgns.state_dict(), os.path.join(args.save_dir, '{}.pt'.format(args.name)))
    # torch.save(optim.state_dict(), os.path.join(args.save_dir, '{}.optim.pt'.format(args.name)))

def only_train_gc(args):

    args.load_model = 100
    d = common_prepare(args)
    idx2word, word2idx, model, sgns, gc_i, gc_o, gc_loss, gc_iter, test_ids, test_labels, optim, criterion, optim_D, beta, log_dir = d.lst

    vocab_size = len(idx2word)
    print("vocab_size", vocab_size, "gendered_neg_size", len(
        gc_loss.neg_ids), "gendered_pos_size", len(gc_loss.pos_ids))

    acc = 0.
    cnt = 0
    for epoch in range(1000):
        pbar = tqdm(get_pos_neg_iter(gc_loss.neg_ids_train,
                                     gc_loss.pos_ids_train, batch_size=256, repeat=False))
        pbar.set_description("[Epoch {}]".format(epoch))
        for batch in pbar :
            ids, labels = batch
            labels = torch.LongTensor(labels).cuda()

            gci_acc = (torch.argmax(gc_i(model.forward_i(test_ids)), dim=-1) == test_labels).float().mean()
            
            Dloss1 = criterion(gc_i(model.forward_i(ids)), labels)
            Dloss2 = criterion(gc_o(model.forward_o(ids)), labels)
            Dloss = Dloss1 + Dloss2
            optim_D.zero_grad()
            Dloss.backward()
            optim_D.step()


            pbar.set_postfix(dict(#acci=acci.item(),
                                #acco=acco.item(),
                                acci=gci_acc.item(),
                                ## gendered words percentage or signal rate
                                sr=float(gc_loss.effect_words) / (gc_loss.tot_words +1e-5),
                                ## pos gendered words percentage or positve rate
                                pr=float(gc_loss.pos_words) / (gc_loss.effect_words+1e-5),
                                Dloss=Dloss.item()))

        #print(output[:32], labels[:32])
        



def eval_gender_classifier(args):
    d = common_prepare(args)
    idx2word, word2idx, model, sgns, gc_i, gc_o, gc_loss, gc_iter, test_ids, test_labels, optim, criterion, optim_D, beta, log_dir = d.lst

    acc = 0.
    cnt = 0
    for batch in tqdm(get_pos_neg_iter(gc_loss.neg_ids, gc_loss.pos_ids, batch_size=512, repeat=False)):
        ids, labels = batch
        labels = torch.LongTensor(labels).cuda()
        output = gc_i(model.forward_i(ids))
        #print(output[:32], labels[:32])
        _acc = (torch.argmax(output, dim=-1) == labels).float().mean()
        acc += _acc
        cnt += 1
    print("Accuracy %.3f" % (acc / cnt))

def dump_embeddings_txt(emb_matrix,emb_name,idx2word):
    with open(emb_name,'w') as outfile:
        print("%d %d" % emb_matrix.shape,file=outfile)
        for idx,wd in enumerate(idx2word):
            try:
                print(wd," ".join([str(x) for x in emb_matrix[idx,:]]),file=outfile)
            except:
                pass
        #



if __name__ == '__main__':
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "dev" or args.mode == "test":
        eval(args.mode)
    elif args.mode == "evalgc":
        eval_gender_classifier(args)
    elif args.mode == "traingc":
        only_train_gc(args)


