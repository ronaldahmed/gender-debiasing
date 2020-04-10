
import warnings
warnings.filterwarnings('ignore')

import glob
import time
import math

import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon.utils import download

import gluonnlp as nlp
import argparse

from data_loader import *

import pdb


def train(args):
  loader = DataLoader(args.dataset,args.batch_size)

  for ep in range(args.epochs):
  	for i, (_input, _target) in enumerate(loader.get_batch("train")):
  		pdb.set_trace()
  		print("-->")





if __name__=="__main__":
	parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", "-d", type=str, help="LM benchmark", default="wikitext-2")
  parser.add_argument("--mode", "-m", type=str, help="Mode [train,eval]", default="train")
  parser.add_argument("--batch_size", "-bs", type=int, help="batch size", default=20)
  parser.add_argument("--epochs", "-ep", type=int, help="batch size", default=50)
	parser.add_argument("-gpu", action="store_true")
  args = parser.parse_args()


  if args.mode=="train":
  	train(args)