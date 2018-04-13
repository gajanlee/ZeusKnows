#!/usr/bin/python
# -*- coding: utf-8 -*-

from __init__ import *
import os
import torch

class Corpus(object):
    """
    文本预处理，获取词汇表，并将字符串文本转换为数字序列。
    """

    def __init__(self, path):
        self.dictionary = vocabulary; vocabulary.add_word("<eop>")
        self.train = self.tokenize(path)

    def tokenize(self, path):
        """文本符号化，转换为数字id表示。"""
        assert os.path.exists(path)

        # 将新词加入到词汇表中
        with open(path, 'r', encoding='utf-8') as f:
            tokens = []
            for i, line in enumerate(f, 1):
                if i % 1000 == 0: logger.info(i)
                line = line.strip("\n").strip(" ").split(" ")
                if len(line) == 0: continue
                
                tokens += [vocabulary.getVocabID(w) for w in line+["<eop>"]]
            return torch.LongTensor(tokens)
                
    def __repr__(self):
        return "Corpus length: %d, Vocabulary size: %d" % (self.train.size(0), len(self.dictionary))


