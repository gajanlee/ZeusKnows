from __init__ import *

class PassageTag:

    def __init__(self):
        self.positive, self.negative = [], []

    def _pro_data(self, d_json):
        if not d_json["match_scores"]: return
        for doc in d_json["documents"]:
            if not doc["is_selected"]:
                self.negative.extend(doc["segmented_paragraphs"])
            elif doc["is_selected"] and len(doc["segmented_paragraphs"]) == 1:
                self.positive.extend(doc["segmented_paragraphs"])

    def start(self, filename=Params.train_files[0]):
        with open(filename) as f:
            for i, l in enumerate(f):
                if i == 100: break
                self._pro_data(json.loads(l))
        self.save()        

    def save(self):
        with open("tag.stat", "w") as f, open("tag_id.stat", "w") as f_id:
            [f.write(json.dumps({"segmented_paragraph": pos, "tag": 1}, ensure_ascii=False)) for pos in self.positive]
            [f.write(json.dumps({"segmented_paragraph": neg, "tag": 0}, ensure_ascii=False)) for neg in self.negative]

            [f_id.write(json.dumps({"segmented_paragraph": [vocabulary.getVocabID(word) for word in pos], "tag": 1})) for pos in self.positive]
            [f_id.write(json.dumps({"segmented_paragraph": [vocabulary.getVocabID(word) for word in neg], "tag": 0})) for neg in self.negative]
        
        logger.info("positive sum is %s" % len(self.positive))
        logger.info("negative sum is %s" % len(self.negative))

    
if __name__ == "__main__":
    PassageTag().start()
