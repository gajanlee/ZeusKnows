#!/usr/bin/python3
from __future__ import division
from __init__ import *
import random

class Writer:
    """Write Infomation to nominated files."""
    # multiple permission writer and created
    __TEXT_FLAG, __TEXT_ID_FLAG, __TEXT_WELL_FLAG, __TEXT_TEST_FLAG = 0x100, 0x010, 0x001, 0x002

    def __init__(self, mode, save_mode=0x113):
        self.mode = mode; self.save_mode = save_mode
        
        self.passage_id = 0
        if self.__permission(self.__TEXT_FLAG):
            self.writers = {
                "DESCRIPTION": open("description_{}.stat".format(mode), "w"),
                "YES_NO": open("yes_no_{}.stat".format(mode), "w"),
                "ENTITY": open("entity_{}.stat".format(mode), "w"),}
        if self.__permission(self.__TEXT_ID_FLAG):        
            self.writers_id = {
                "DESCRIPTION": open("description_id_{}.stat".format(mode), "w"),
                "YES_NO": open("yes_no_id_{}.stat".format(mode), "w"),
                "ENTITY": open("entity_id_{}.stat".format(mode), "w"),}
        if self.__permission(self.__TEXT_WELL_FLAG):
            self.writers_format = {
                "DESCRIPTION": open("description_wellformat_{}.stat".format(mode), "w"),
                "YES_NO": open("yes_no_wellformat_{}.stat".format(mode), "w"),
                "ENTITY": open("entity_wellformat_{}.stat".format(mode), "w"),}
        if self.__permission(self.__TEXT_TEST_FLAG):    # generate test data format
            self.writers_test = {
                "DESCRIPTION": open("description_rank_{}.stat".format(mode), "w"),
                "YES_NO": open("yes_no_rank_{}.stat".format(mode), "w"),
                "ENTITY": open("entity_rank_{}.stat".format(mode), "w"),}
            self.writers_test_id = open("rank_id.stat", "w")
       
        
        #self.train_stat_writer = open("dev_stat.info", "w")
        #self.stat = {}
        # for key in ["DESCRIPTION", "YES_NO", "ENTITY", "DESCRIPTION_question", "YES_NO_question", "ENTITY_question"]:
        #    self.stat[key] = [0] * 100000
    
    def __permission(self, req_mode):
        return req_mode & self.save_mode

    def write(self, data):
        if self.__permission(self.__TEXT_FLAG):
            self.writers[data["question_type"]].write(json.dumps(data, ensure_ascii=False) + "\n")
        #self.stat[data["question_type"]][len(data["segmented_paragraph"])] += 1
        #self.stat[data["question_type"] + "_question"][len(data["segmented_question"])] += 1

    def write_id(self, data):
        if self.__permission(self.__TEXT_ID_FLAG):
            self.writers_id[data["question_type"]].write(json.dumps(data) + "\n")

    def write_wellformat(self, data):
        if not self.__permission(self.__TEXT_WELL_FLAG): return
        if data["question_type"] == "YES_NO":
            for c, label in zip(data["segmented_answers"], data["yesno_answers"]):
                self.writers_format["YES_NO"].write(
                    "%s __label__%s\n" % (" ".join(c), label)
                )

    def write_test(self, data, id=False):
        if id == True:
            self.writers_test_id.write(json.dumps(data) + "\n")
        else:
            self.writers_test[data["question_type"]].write(json.dumps(data) + "\n")

    def close(self, signal=0x111):
        if self.__permission(self.__TEXT_FLAG):
            [writer.close() for writer in self.writers.values()] 
        if self.__permission(self.__TEXT_ID_FLAG):
            [writer.close() for writer in self.writers_id.values()]
        if self.__permission(self.__TEXT_WELL_FLAG):
            [writer.close() for writer in self.writers_format.values()]
        if self.__permission(self.__TEXT_TEST_FLAG):
            [writer.close() for writer in self.writers_test.values()]
            self.writers_test_id.close()

    def preprocess(self):
        with open(Params.data_files_format.format(mode=self.mode)) as fp:
            if self.mode == "test":
                [self.test_process(json.loads(line), i) for i, line in enumerate(fp)]
                #[self.test_process(json.loads(line), i) for i, line in enumerate(fp)]
            elif self.mode in ["train", "dev"]:
                [self.train_process(json.loads(line), i) for i, line in enumerate(fp)]
    
    def train_process(self, data_json, i):
        if i % 1000 == 0: logger.info(self.mode.upper() + " %s LINE..." % i)

        if self.mode == "test":
            return
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
        
        base_format["segmented_answers"] = data_json["segmented_answers"]
        if data_json["question_type"] == "YES_NO":
            base_format["yesno_answers"] = data_json["yesno_answers"]   # may be more than one

        self.write_wellformat(base_format)

    def test_process(self, data_json, i):
        print(i)
        format = {
            "question_id": data_json["question_id"],
            "question_type": data_json["question_type"],
            "segmented_question": data_json["segmented_question"],
            "char_question": [vocabulary.getCharID(word, True) for word in data_json["segmented_question"]]
        }
        count = 0
        for doc in data_json["documents"]:
            rank = doc["bs_rank_pos"]
            for para in doc["segmented_paragraphs"]:
                if count >= 3: return
                if random.randint(0, 2) != 0: continue
                count += 1
                format["rank"] = rank
                format["passage_id"] = self.passage_id
                self.passage_id += 1
                format["segmented_paragraph"] = para
                format["char_paragraph"] = [vocabulary.getCharID(word, True) for word in para]
                self.write_test(format, False)
                format["segmented_paragraph"] = [vocabulary.getVocabID(id) for id in para]
                format["segmented_question"] = [vocabulary.getVocabID(id) for id in data_json["segmented_question"]]
                self.write_test(format, True)

    def __enter__(self):
        return self
    def __exit__(self, *exc_info):
        self.close(self.__TEXT_WELL_FLAG)

if __name__ == "__main__":
    with Writer("test", 0x002) as wt:
        wt.preprocess()

    """with Writer("train", 0x001) as wt, Writer("dev", 0x001) as wd:
        logger.info("Start process trainset...")
        wt.preprocess()
        logger.info("Start process devset...")
        wd.preprocess()"""
