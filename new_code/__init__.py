import re
import json
from params import Params

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
        self.vocab_list, self.charlst, self.wordlst = {"unknown": 0}, {"<unknown>": 0}, []
        self.vocab_ids = self.char_ids = 1
        self.load_char_dict()

    def process_word_list(self, lst):
        self.wordlst.append(lst)
        for word in lst:
            if word not in self.vocab_list and self.unfilter(word):
                self.vocab_list[word] = self.vocab_ids; self.vocab_ids += 1
    
    def unfilter(self, word):
        return False if len(word) > 5 or re.findall(re.compile(r'[a-zA-Z0-9]'), word) or self.notcommon(word) else True

    def notcommon(self, word):
        for char in word:
            if char not in self.charlst:
                return True
        return False

    def load_char_dict(self):
        with open(Params.char_dict_path) as fp:
            for id, line in enumerate(fp, 1):
                self.charlst[line[0]] = id
    
    def getVocabID(self, word):
        return self.vocab_list.get(word, 0)

    def save(self):
        # write and close
        fp1, fp2 = open(Params.wordlst_path, "w"), open("_id"+Params.wordlst_path, "w")
        for words in self.wordlst:
            fp1.write("%s\n" % " ".join(words))
            fp2.write("%s\n" % " ".join([self.getVocabID(vocab) for vocab in words]))
        fp1.close(); fp2.close()
        with open(Params.vocab_path, "w") as fp:
            for word, id in self.vocab_list.items():
                fp.write("%s %s" % (word, id))
        with open(Params.char_path, "w") as fp:
            for char, id in self.charlst.items():
                fp.write("%s %s" % (char, id))
        

vocabulary = Vocabulary()