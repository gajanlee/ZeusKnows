# -*- coding: utf-8 -*-
#/usr/bin/python2
from __future__ import print_function

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')   

class Params():

    # data
    data_size = -1 # -1 to use all data
    num_epochs = 3
    train_prop = 0.9 # Not implemented atm
    data_dir = "../"
    
    tp = "search"
    train_dir = data_dir + "{}.train.net.json".format(tp)
    dev_dir = "../{}.dev.net.json".format(tp)
    test_rank_dir = data_dir + "rank_id.stat"
    #train_dir = data_dir + "entity_id_train.stat"
    #dev_dir = data_dir + "entity_id_dev.stat"
    #logdir = "./train/train"
    logdir = "./train/{}".format(tp)
    outputdir = "../res/{}.res".format(tp)
    logdir_rank = "./rank/train"
    glove_dir = "./glove.840B.300d.txt" # Glove file name (If you want to use your own glove, replace the file name here)
    glove_char = "./glove.840B.300d.char.txt" # Character Glove file name
    coreNLP_dir = "./stanford-corenlp-full-2017-06-09" # Directory to Stanford coreNLP tool

    # Data dir
    target_dir = "indices.txt"
    q_word_dir = "words_questions.txt"
    q_chars_dir = "chars_questions.txt"
    p_word_dir = "words_context.txt"
    p_chars_dir = "chars_context.txt"

    # Training
	# NOTE: To use demo, put batch_size == 1
    #mode = "gen_rank" # case-insensitive options: ["train", "test", "debug"]
    mode = "demo"
    #mode = "train"
    dropout = 0.2 # dropout probability, if None, don't use dropout
    zoneout = None # zoneout probability, if None, don't use zoneout
    optimizer = "adam" # Options: ["adadelta", "adam", "gradientdescent", "adagrad"]
    batch_size = 50 if mode is not "test" else 100# Size of the mini-batch for training
    batch_size = 5 if mode is "demo" else batch_size    # demo generates 5 result answers
    save_steps = 50 # Save the model at every 50 steps
    clip = True # clip gradient norm
    norm = 5.0 # global norm
    # NOTE: Change the hyperparameters of your learning algorithm here
    opt_arg = {'adadelta':{'learning_rate':1, 'rho': 0.95, 'epsilon':1e-6},
                'adam':{'learning_rate':1e-3, 'beta1':0.9, 'beta2':0.999, 'epsilon':1e-8},
                'gradientdescent':{'learning_rate':1},
                'adagrad':{'learning_rate':1}}

    # Architecture
    SRU = True # Use SRU cell, if False, use standard GRU cell
    max_p_len = 450 #300 # Maximum number of words in each passage context
    max_q_len = 20 #30 # Maximum number of words in each question context
    max_char_len = 5 #16 # Maximum number of characters in a word
    vocab_size = 163825 #91604 # Number of vocabs in glove.840B.300d.txt + 1 for an unknown token
    char_vocab_size = 7057 #95 # Number of characters in glove.840B.300d.char.txt + 1 for an unknown character
    emb_size = 300 # Embeddings size for words
    char_emb_size = 30 #8 # Embeddings size for characters
    attn_size = 75 # RNN cell and attention module size
    num_layers = 3 # Number of layers at question-passage matching
    bias = True # Use bias term in attention


    if mode == "prank":
        train_dir = data_dir + "tag_id_train.stat"
        dev_dir = "tag_id_dev.stat"
        test_dir = "rank_id.stat"

import json
class Vocabulary:
    def __init__(self):
        self.vocab_dict, self.char_dict = {}, {}
        self.load_vocab_dict("../vocab.dict")
        self.load_char_dict("../char.dict")
        print("vocab_dict size is ", len(self.vocab_dict))
        print("char_dict size is ", len(self.char_dict))

    def load_vocab_dict(self, vocab_path):
        vocab_dict = json.load(open(vocab_path))
        for vocab, _id in vocab_dict.items():
            self.vocab_dict[_id] = vocab
        print("=======>", self.vocab_dict[100])

        #with open(vocab_path) as fp: 
            #for i, line in enumerate(fp):
                #self.vocab_dict[i] = line.split(" ")[0] 
                #self.vocab_dict[res[1]] = int(res[0])

    def load_char_dict(self, char_path):
        char_dict = json.load(open(char_path))
        for char, _id in char_dict.items():
            self.char_dict[_id] = char

    def ind2word(self,ids):
        output = []
        for i in ids:
            output.append(self.vocab_dict[i])
        return " ".join(output)

    def ind2char(self,ids):
        output = []
        for i in ids:
            for j in i:
                output.append(self.char_dict[j])
            output.append(" ")
        return "".join(output)


