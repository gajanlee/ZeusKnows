from __init__ import *

class PassageTag:

    __NEGATIVE, __POSITIVE = 0, 1

    def __init__(self):
        self.attr = [[], []]
        # self.positive, self.negative = [], []

    def _pro_data(self, d_json):
        if not d_json["match_scores"] or (len(d_json["segmented_question"]) == 0): return
        for doc in d_json["documents"]:
            if not doc["is_selected"]:
                self.attr[self.__NEGATIVE].extend(
                    [{
                        "segmented_paragraph": seg_par,
                        "segmented_question": d_json["segmented_question"],
                        "tag": self.__NEGATIVE,
                        "char_paragraph": [[char for char in word] for word in seg_par],
                        "char_question": [[char for char in word] for word in d_json["segmented_question"]],
                    } for seg_par in doc["segmented_paragraphs"]]
                )
            elif doc["is_selected"] and len(doc["segmented_paragraphs"]) == 1:
                self.attr[self.__POSITIVE].extend(
                    [{
                        "segmented_paragraph": seg_par,
                        "segmented_question": d_json["segmented_question"],
                        "tag": self.__POSITIVE,
                        "char_paragraph": [[char for char in word] for word in seg_par],
                        "char_question": [[char for char in word] for word in d_json["segmented_question"]],
                    } for seg_par in doc["segmented_paragraphs"]]
                )

    def start(self, filename=Params.train_files[0]):
        with open(filename) as f:
            for i, l in enumerate(f, 1):
                if i % 1000 == 0: logger.info("%s / 90000" % i )
                self._pro_data(json.loads(l))
        self.save()

    def save(self):
        with open("tag_train.stat", "w") as f, open("tag_id_train.stat", "w") as f_id:
            import random
            self.attr[self.__NEGATIVE] = list(filter(lambda x: random.randint(1, 8) <= 2, self.attr[self.__NEGATIVE]))
            [f.write("\n".join([json.dumps(attr, ensure_ascii=False) for attr in attrs])) for attrs in self.attr]

            for attrs in self.attr:
                for attr in attrs:
                    attr["segmented_paragraph"] = [vocabulary.getVocabID(word) for word in attr["segmented_paragraph"]]
                    attr["segmented_question"] = [vocabulary.getVocabID(word) for word in attr["segmented_question"]]
                    attr["char_paragraph"] = [[vocabulary.getCharID(char) for char in word] for word in attr["char_paragraph"]]
                    attr["char_question"] = [[vocabulary.getCharID(char) for char in word] for word in attr["char_question"]]
            
            for attrs in self.attr:
                f_id.write("\n".join([json.dumps(attr) for attr in attrs]))
                f_id.write("\n")
            
        logger.info("positive sum is %s" % len(self.attr[self.__POSITIVE]))
        logger.info("negative sum is %s" % len(self.attr[self.__NEGATIVE]))

    
if __name__ == "__main__":
    PassageTag().start()
