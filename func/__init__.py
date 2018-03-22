import re
import json
from params import Params
import logging

logger = logging.getLogger("vocab_logger")
logging.basicConfig(level = logging.DEBUG)

class Word:
    def __init__(self, id, content):
        self.count = 1
        self.content = content
        self.id = id
        self.length = len(content)

    def inc(self):
        self.count += 1

    def __str__(self):
        return "<Word id=%s content=%s count=%s>" % (self.id, self.content, self.count)
    
    def __repr__(self):
        return "<Word id=%s content=%s count=%s>" % (self.id, self.content, self.count)


class Vocabulary:
    def __init__(self):
        self.vocab_dict, self.char_dict, self.wordlst = {"unknown": [0, 0]}, {"<unknown>": [0, 0]}, []
        self.vocab_ids = self.char_ids = 1
        #self.load_char_dict()

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

    def notcommon(self, word):
        return False
        for char in word:
            if (char not in self.char_dict and char.isalpha()):
                print(char, char.isalpha())
                return True
        return False

    def load_char_dict(self):
        with open(Params.char_dict_path) as fp:
            for id, line in enumerate(fp, 1):
                self.char_dict[line[0]] = id
        logger.info("Char dictionary loaded DONE! SUM %s ." % len(self.char_dict))
    
    def getVocabID(self, word):
        return self.vocab_dict.get(word, 0)

    def save(self):
        # write and close
        fp1, fp2 = open(Params.wordlst_path, "w"), open("_id"+Params.wordlst_path, "w")
        for words in self.wordlst:
            fp1.write("%s\n" % " ".join(words))
            fp2.write("%s\n" % " ".join([str(self.getVocabID(vocab)) for vocab in words]))
        fp1.close(); fp2.close()

        
        vocab_lst = sorted(self.vocab_dict.items(), key=lambda x: x[1][1])
        char_lst = sorted(self.char_dict.items(), key=lambda x: x[1][1])
        print(self.char_dict["五"])
        print(self.char_dict["花"])
        with open(Params.vocab_path, "w") as fp:
            for st in vocab_lst:
                fp.write("%s %s %s\n" % (st[0], st[1][0], st[1][1]))
        with open(Params.char_path, "w") as fp:
            for st in char_lst:
                fp.write("%s %s %s\n" % (st[0], st[1][0], st[1][1]))
        

vocabulary = Vocabulary()
