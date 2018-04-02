import json
class QPair:
    def __init__(self):
        self.info = {}
        for mode in ["yes_no", "entitiy", "description"]:
            with open("{}_rank_test.stat".format(mode)) as f:
                for line in f:
                    data = json.loads(line)
                    ap = {
                        "passage_id": data["passage_id"],
                        "question_type": data["question_type"],
                        "segmented_paragraph": data["segmented_paragraph"]
                    }
                    if data["question_id"] not in self.info:
                        data["question_id"] = [ap]
                    else:
                        data["question_id"].append(ap)
        
        self.res = {}
        fw = open("upload_res.json", "w")
        with open("test_res.stat") as f:
            for line in f:
                data = json.loads(line)
                for que in self.info[data["question_id"]]:
                    if que["passage_id"] == data["question_id"]:
                        if data["question_id"] not in self.res:
                            self.res[data["question_id"]] = {
                                "question_id": data["question_id"],
                                "question_type": que["question_type"],
                                "answers": "".join(que["segmented_paragraph"][data["spans"][0]: data[spans][1]+1]),
                                "yesno_answers": [],
                                "entity_answers": [[]],
                            }
                        fw.write(json.dumps({
                            self.res[data["question_id"]]
                        }) + "\n")
        fw.close()

if __name__ == "__main__":
    QPair()
            
