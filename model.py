# -*- coding: utf-8 -*-

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from torch import LongTensor as LT
from torch import IntTensor as IT
from torch import FloatTensor as FT

from semi_annotate.get_gendered_word import get_gendered_words
class Bundler(nn.Module):

    def forward(self, data):
        raise NotImplementedError

    def forward_i(self, data):
        raise NotImplementedError

    def forward_o(self, data):
        raise NotImplementedError


class Word2Vec(Bundler):

    def __init__(self, vocab_size=20000, embedding_size=300, padding_idx=0):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.ivectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ivectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ovectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True

    def forward(self, data):
        return self.forward_i(data)

    def forward_i(self, data):
        v = LT(data)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        return self.ivectors(v)

    def forward_o(self, data):
        v = LT(data)
        v = v.cuda() if self.ovectors.weight.is_cuda else v
        return self.ovectors(v)


class SGNS(nn.Module):

    def __init__(self, embedding, vocab_size=20000, n_negs=20, weights=None):
        super(SGNS, self).__init__()
        self.embedding = embedding
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        self.weights = None
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)

    def forward(self, iword, owords):
        batch_size = iword.size()[0]
        context_size = owords.size()[1]
        if self.weights is not None:
            nwords = t.multinomial(self.weights, batch_size * context_size * self.n_negs, replacement=True).view(batch_size, -1)
        else:
            nwords = FT(batch_size, context_size * self.n_negs).uniform_(0, self.vocab_size - 1).long()
        ivectors = self.embedding.forward_i(iword).unsqueeze(2)
        ovectors = self.embedding.forward_o(owords)
        nvectors = self.embedding.forward_o(nwords).neg()
        oloss = t.bmm(ovectors, ivectors).squeeze().sigmoid().log().mean(1)
        nloss = t.bmm(nvectors, ivectors).squeeze().sigmoid().log().view(-1, context_size, self.n_negs).sum(2).mean(1)
        return -(oloss + nloss).mean()

class GenderClassifier(nn.Module):
    def __init__(self, word_dim, hidden_units = None):
        super().__init__()
        self.word_dim = word_dim
        if hidden_units is None:
            hidden_units = word_dim * 4
        self.dense1 = nn.Linear(word_dim, hidden_units)
        self.dense2 = nn.Linear(hidden_units, 2)
        self.act = nn.ReLU()
    
    def forward(self, input):
        hidden = self.dense1(input)
        hidden = self.act(hidden)
        output = self.dense2(hidden)
        return output

def softmax_cross_entropy(input, target, mask = None):
    #print(input.shape, target.shape, mask.shape)
    logp = - F.log_softmax(input, dim=1)
    target = target.unsqueeze(1)
    mask = mask.unsqueeze(1)
    logpy = t.gather(logp, 1, target).view(-1)
    if mask is None:
        logpy =  logpy.mean()
    else:
        #mask = FT(mask)
        logpy = (logp * mask).mean()
    return logpy


def softmax_cross_entropy_uniform(input):
    #print(input.shape, target.shape, mask.shape)
    logp = - F.log_softmax(input, dim=1)
    classes = list(input.size())[1]
    uni_dist = t.ones(size=(1,classes), dtype = t.float32).cuda() / classes

    logpy = -logp * uni_dist
    logpy = logpy.mean()
    return logpy

def t_size(tensor):
    prod = 1
    for i in list(tensor.size()):
        prod *= i
    return prod

class GetGenderLoss(nn.Module):
    def __init__(self, embedding, iclassifier, oclassifier, word2idx):
        super().__init__()
        self.iclassifier = iclassifier
        self.oclassifier = oclassifier
        self.embedding = embedding
        ## This is gendered words, approx ~ 120k words pos v neg
        neg_words, pos_words = get_gendered_words()
        self.neg_ids = set()
        self.pos_ids = set()
        self.gendered_ids = set()
        for _w in neg_words:
            if _w in word2idx:
                self.neg_ids.add(word2idx[_w])
                self.gendered_ids.add(word2idx[_w])
        for _w in pos_words:
            if _w in word2idx:
                self.pos_ids.add(word2idx[_w])
                self.gendered_ids.add(word2idx[_w])
        self.criterion = softmax_cross_entropy
        self.effect_words = 0
        self.tot_words = 0

    def get_label(self, word):
        if word in self.gendered_ids:
            if word in self.neg_ids:
                return 0
            elif word in self.pos_ids:
                return 1
        else:
            return 2

    def prepare_words(self, words):
        words = words.view(-1)
        labels = []
        masks = []
        for word in words.tolist():
            _label = self.get_label(word)
            if _label==2:
                masks.append(0.)
            else:
                masks.append(1.)
                self.effect_words += 1
            labels.append(_label)

        labels = LT(labels).cuda()
        masks = IT(masks).cuda()
        return words, labels, masks

    def forward_D(self, iword, owords, acc = False):
        #print(iword.shape)
        #print(owords.shape)
        iwords, ilabels, imasks = self.prepare_words(iword)
        ilogits = self.iclassifier(self.embedding.forward_i(iwords))
        loss = self.criterion(ilogits, ilabels, imasks)
        if acc:
            max_index = ilogits.argmax(dim=1)
            _acc1 = ((max_index.eq(ilabels)).float() *
                    imasks).sum() / imasks.sum().float()
        owords, olabels, omasks = self.prepare_words(owords)
        ologits = self.oclassifier(self.embedding.forward_o(owords))
        loss += self.criterion(ologits, olabels, omasks)
        self.tot_words += t_size(owords) + t_size(iword)
        if acc:
            max_index = ologits.argmax(dim=1)
            _acc2 = ((max_index.eq(olabels)).float() *
                     omasks).sum() / omasks.sum().float()
        if acc:
            return loss, _acc1, _acc2
        else:
            return loss

    def forward_E(self, iword, owords):
        ## Note that for embedding, its goal is to neither let classifier output positive gender nor negative gender
        #print(iword.shape)
        #print(owords.shape)
        iwords, _, _ = self.prepare_words(iword)
        ilogits = self.iclassifier(self.embedding.forward_i(iwords))
        loss = softmax_cross_entropy_uniform(ilogits)


        owords, _, _ = self.prepare_words(owords)
        ologits = self.oclassifier(self.embedding.forward_o(owords))
        loss += softmax_cross_entropy_uniform(ologits)
        self.tot_words += t_size(owords) + t_size(iword)

        return loss

        
