import torch
import nltk

import gluonnlp as nlp

class DataLoader:
  def __init__(self,_dataset,_batch_size,_bptt=35):
    dataset_name = _dataset
    # Load the dataset
    self.train_dataset, self.val_dataset, self.test_dataset = [
      nlp.data.WikiText2(
          segment=segment, bos=None, eos='<eos>', skip_empty=False)
      for segment in ['train', 'val', 'test']
    ]

    # Extract the vocabulary and numericalize with "Counter"
    self.vocab = nlp.Vocab(
      nlp.data.Counter(train_dataset), padding_token=None, bos_token=None)

    self._bptt = 35
    self.batch_size = _batch_size

  def get_batch(self,split="train")
    # Batchify for BPTT
    bptt_batchify = nlp.data.batchify.CorpusBPTTBatchify(
        self.vocab, self.bptt, self.batch_size, last_batch='discard')
    data = []
    if   split=="train":  data = self.train_dataset
    elif split=="val":  data = self.val_dataset
    elif split=="test":  data = self.test_dataset

    return bptt_batchify(data)
  

