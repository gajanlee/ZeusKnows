#!/usr/bin/python3
from __future__ import division
from __init__ import *

class Writer:
    def __init__(self, mode):
        self.mode = mode
        self.writers = {
            "DESCRIPTION": open("description_{}.stat".format(mode), "w"),
            "YES_NO": open("yes_no_{}.stat".format(mode), "w"),
            "ENTITY": open("entity_{}.stat".format(mode), "w"),}
        self.writers_id = {
            "DESCRIPTION": open("description_id_{}.stat".format(mode), "w"),
            "YES_NO": open("yes_no_id_{}.stat".format(mode), "w"),
            "ENTITY": open("entity_id_{}.stat".format(mode), "w"),} 
        
        #self.train_stat_writer = open("dev_stat.info", "w")
        #self.stat = {}
        # for key in ["DESCRIPTION", "YES_NO", "ENTITY", "DESCRIPTION_question", "YES_NO_question", "ENTITY_question"]:
        #    self.stat[key] = [0] * 100000
    
    def write(self, data):
        self.writers[data["question_type"]].write(json.dumps(data, ensure_ascii=False) + "\n")
        #self.stat[data["question_type"]][len(data["segmented_paragraph"])] += 1
        #self.stat[data["question_type"] + "_question"][len(data["segmented_question"])] += 1

    def write_id(self, data):
        self.writers_id[data["question_type"]].write(json.dumps(data) + "\n")

    def close(self, signal=0):
        if signal: return
        [writer.close() for writer in self.writers.values()] 
        [writer.close() for writer in self.writers_id.values()]

    def preprocess(self):
        with open(Params.data_files_format.format(mode=self.mode)) as fp:
            if self.mode == "test":
                pass
            elif self.mode in ["train", "dev"]:
                [self.train_process(json.loads(line), i) for i, line in enumerate(fp)]
    
    def train_process(self, data_json, i):
        if i % 1000 == 0: logger.info(self.mode.upper() + " %s LINE..." % i)
        if not data_json["match_scores"]: return
        answer_spans = data_json["answer_spans"][0] # Actually it only ones
        doc = data_json["documents"][data_json["answer_docs"][0]]
        base_format = {
            "question_type": data_json["question_type"],    # YES_NO / ENTITY / DESCRIPTION$
            "question_id": data_json["question_id"],        # ID(int)$
            "fact_or_opinion": data_json["fact_or_opinion"],# FACT  / OPINION$
            "segmented_paragraph": doc["segmented_paragraphs"][doc["most_related_para"]],  # Now, only one paragraph, shape [para_len]$
            "match_scores": data_json["match_scores"],
            "answer_spans": data_json["answer_spans"],}

        base_format["segmented_paragraph"] = doc["segmented_paragraphs"][doc["most_related_para"]]
        base_format["segmented_answer"] = base_format["segmented_paragraph"][answer_spans[0]:answer_spans[1]]
        base_format["segmented_question"] = data_json["segmented_question"]
        base_format["char_paragraph"] = [vocabulary.getCharID(word, True) for word in base_format["segmented_paragraph"]]
        base_format["char_question"] = [vocabulary.getCharID(word, True) for word in base_format["segmented_question"]]
        self.write(base_format)
        base_format["segmented_paragraph"] = [vocabulary.getVocabID(id) for id in base_format["segmented_paragraph"]]
        base_format["segmented_answer"] = [vocabulary.getVocabID(id) for id in base_format["segmented_answer"]]
        base_format["segmented_question"] = [vocabulary.getVocabID(id) for id in base_format["segmented_question"]]
        self.write_id(base_format)
        
    def __enter__(self):
        return self
    def __exit__(self, *exc_info):
        self.close()

if __name__ == "__main__":
    with Writer("train") as wt, Writer("dev") as wd:
        logger.info("Start process trainset...")
        wt.preprocess()
        logger.info("Start process devset...")
        wd.preprocess()
