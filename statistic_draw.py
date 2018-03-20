#!/usr/bin/python3
from __future__ import division
import json, logging
logger = logging.getLogger("classfication")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--vocabpath", default="./word.dict", help="vocabulary dictionary path")
parser.add_argument("--charpath", default="./word.txt", help="character dictionary path")
args = parser.parse_args()

class Vocabulary:
    def __init__(self):
        self.vocab_dict, self.char_dict = {}, {"<unknown>": 0, 0: "<unknown>"}
        self.load_vocab_dict(args.vocabpath)
        self.load_char_dict(args.charpath)

    def load_vocab_dict(self, vocab_path):
        with open(vocab_path) as fp:
            for line in fp:
                res = line.split(' ')
                self.vocab_dict[res[1]] = int(res[0])
                self.vocab_dict[int(res[0])] = res[1]

    def load_char_dict(self, char_path):
        with open(char_path) as fp:
            for i, line in enumerate(fp, 1):
                self.char_dict[line[:-1]] = i
                self.char_dict[i] = line[:-1]
                #res = line.split(' ')
                #print(res[0], '+', res[1])
                #self.char_dict[res[0]] = int(res[1])
                #self.char_dict[int(res[1])] = res[0]

    def get(self, vocab):
        if vocab not in self.vocab_dict:
            return 0 if type(vocab) == str else '<unknown>'
        return self.vocab_dict[vocab]
    def get_char(self, char):
        if char not in self.char_dict:
            return 0 if type(char) == str else "<unknown>"
        return self.char_dict[char]

    def close(self):
        with open("char2.dict", "w") as fp:
            for k, v in self.char_dict.items():
                if type(k) == str:
                    fp.write("%s %s\n" %(v, k))

class Writer:
    def __init__(self):
        self.writers = {
            "DESCRIPTION": open("description.stat", "w"),
            "YES_NO": open("yes_no.stat", "w"),
            "ENTITY": open("entity.stat", "w"),}
        self.writers_id = {
            "DESCRIPTION": open("description_id.stat", "w"),
            "YES_NO": open("yes_no_id.stat", "w"),
            "ENTITY": open("entity.stat", "w"),} 
        self.train_stat_writer = open("train_stat.info", "w")
        self.stat = {}
        for key in ["DESCRIPTION", "YES_NO", "ENTITY", "DESCRIPTION_question", "YES_NO_question", "ENTITY_question"]:
            self.stat[key] = [0] * 100000
    
    def write(self, data):
        self.writers[data["question_type"]].write(json.dumps(data, ensure_ascii=False) + "\n")
        self.stat[data["question_type"]][len(data["segmented_paragraph"])] += 1
        self.stat[data["question_type"] + "_question"][len(data["segmented_question"])] += 1

    def write_id(self, data):
        self.writers_id[data["question_type"]].write(json.dumps(data) + "\n")

    def close(self, signal=0):
        if signal: return
        [writer.close() for writer in self.writers.values()] 
        [writer.close() for writer in self.writers_id.values()]
        self.temp_stat = {}
        for key, value in self.stat.items():
            self.temp_stat[key] = {}
            total = temp_total = 0
            for l, v in enumerate(value):
                total += v
            for l, v in enumerate(value):
                temp_total += v
                for threshold in [0.8, 0.85, 0.90, 0.95, 0.97, 0.99]:
                    if temp_total / total >= threshold and self.temp_stat[key].get(threshold, None) == None:
                        self.temp_stat[key][threshold] = l
        self.train_stat_writer.write(json.dumps(self.temp_stat))
        self.train_stat_writer.close()

def preprocess(mode):
    with open("../DuReader/data/preprocessed/{mode}set/zhidao.{mode}.json".format(mode=mode)) as fp:
        if mode == "test":
            [test_process(json.loads(line)) for line in fp]
        elif mode == "train":
            [train_process(i, json.loads(line)) for i, line in enumerate(fp)]

def test_process(l):
    pass

writer = Writer()
vocabulary = Vocabulary()
def train_process(lineth, data_json):
    answer_spans = data_json["answer_spans"]
    if not data_json["match_scores"]: return
    doc = data_json["documents"][data_json["answer_docs"][0]]
    base_format = {
        "question_type": data_json["question_type"],    # YES_NO / ENTITY / DESCRIPTION$
        "question_id": data_json["question_id"],        # ID(int)$
        "fact_or_opinion": data_json["fact_or_opinion"],# FACT  / OPINION$
        "segmented_paragraph": doc["segmented_paragraphs"][doc["most_related_para"]],  # Now, only one paragraph, shape [para_len]$
        "match_scores": data_json["match_scores"],
        "answer_spans": data_json["answer_spans"][0],}
    
    base_format["segmented_paragraph"] = doc["segmented_paragraphs"][doc["most_related_para"]]
    base_format["segmented_answer"] = base_format["segmented_paragraph"][answer_spans[0][0]:answer_spans[0][1]]
    base_format["segmented_question"] = data_json["segmented_question"]
    base_format["char_paragraph"] = [[vocabulary.get_char(char) for char in word] for word in base_format["segmented_paragraph"]]
    base_format["char_question"] = [[vocabulary.get_char(char) for char in word] for word in base_format["segmented_question"]]
    writer.write(base_format)
    base_format["segmented_paragraph"] = [vocabulary.get(id) for id in base_format["segmented_paragraph"]]
    base_format["segmented_answer"] = [vocabulary.get(id) for id in base_format["segmented_answer"]]
    base_format["segmented_question"] = [vocabulary.get(id) for id in base_format["segmented_question"]]
    writer.write_id(base_format)
    

if __name__ == "__main__":
    preprocess("train")
    writer.close()
    vocabulary.close()
