import re, os
import json
from params import Params
import logging

logger = logging.getLogger("vocab_logger")
logging.basicConfig(level = logging.DEBUG, 
                    format = "%(asctime)s : %(levelname)s  %(message)s",
                    datefmt = "%Y-%m-%d %A %H:%M:%S")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--endline")
args = parser.parse_args()

class Vocabulary:
    def __init__(self):
        self.vocab_dict, self.char_dict, self.wordlst = {"unknown": [0, Params.count_threshold]}, {"<unknown>": [0, Params.count_threshold]}, []
        self.vocab_ids = self.char_ids = 1
        #self.load_char_dict()
        if os.path.exists(Params.vocab_path): self.load_vocab_dict()
        if os.path.exists(Params.char_path): self.load_char_dict()

    def process_word_list(self, lst):
        self.wordlst.append(lst)
        for word in lst:
            if self.unfilter(word):
                if word not in self.vocab_dict: self.vocab_dict[word] = [self.vocab_ids, 1]; self.vocab_ids += 1
                else: self.vocab_dict[word][1] += 1
            for char in word:
                if char not in self.char_dict:
                    self.char_dict[char] = [self.char_ids, 1]; self.char_ids += 1
                else:
                    self.char_dict[char][1] += 1
    
    def unfilter(self, word):
        return False if len(word) > 5 or re.findall(re.compile(r'[a-zA-Z0-9]'), word) or self.notcommon(word) else True

    # judge if the char is common.open
    # Update: We choose characters by displayed count.
    def notcommon(self, word):
        return False
        for char in word:
            if (char not in self.char_dict and char.isalpha()):
                print(char, char.isalpha())
                return True
        return False

    # Load Char Dict As Writer Formate: content, id, count
    def load_char_dict(self):
        with open(Params.char_path) as fp:
            for line in fp:
                res = line.split(" ")
                try: self.char_dict[res[0]] = int(res[1])
                except: pass
        logger.info("Char dictionary loaded DONE! SUM %s ." % len(self.char_dict))

    # Get A Char's ID
    def getCharID(self, char):
        return self.char_dict.get(char, 0)

    # Format as same as char dict.
    def load_vocab_dict(self):
        with open(Params.vocab_path) as fp:
            for i, line in enumerate(fp):
                res = line.split(" ")
                try:self.vocab_dict[res[0]] = int(res[1])
                except: pass
        logger.info("Vocabulary dictionary loaded DONE! SUM %s ." % len(self.vocab_dict))        

    def getVocabID(self, word):
        return self.vocab_dict.get(word, 0)

    def purify_sorted(self, data_dict):
        sort_list = sorted(list(filter(lambda x: x[1][1] >= Params.count_threshold, data_dict.items())), key=lambda x: x[1][1])
        res_list = []; data_dict.clear()
        for id, ele in enumerate(sort_list):
            res_list.append([ele[0], [id, ele[1][1]] ])
            data_dict[ele[0]] = id
        return res_list
    
    def save(self):
        vocab_lst = self.purify_sorted(self.vocab_dict)
        char_lst  = self.purify_sorted(self.char_dict)
        # write and close
        fp1, fp2 = open(Params.wordlst_path, "w"), open("_id"+Params.wordlst_path, "w")
        for words in self.wordlst:
            fp1.write("%s\n" % " ".join(words))
            fp2.write("%s\n" % " ".join([str(self.getVocabID(vocab)) for vocab in words]))
        fp1.close(); fp2.close()

        #vocab_lst = sorted(self.vocab_dict.items(), key=lambda x: x[1][1])
        #char_lst = sorted(self.char_dict.items(), key=lambda x: x[1][1])
        with open(Params.vocab_path, "w") as fp:
            for st in vocab_lst:
                fp.write("%s %d %d,\n" % (st[0], st[1][0], st[1][1]))
        with open(Params.char_path, "w") as fp:
            for st in char_lst:
                fp.write("%s %d %d,\n" % (st[0], st[1][0], st[1][1]))
        

vocabulary = Vocabulary()
