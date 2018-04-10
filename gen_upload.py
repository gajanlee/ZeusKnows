import json

lookup = {}
with open("search/total_test.json") as t:
    for i, line in enumerate(t, 1):
        data = json.loads(line)
        if data["question_id"] in lookup:
            lookup[data["question_id"]]["paragraphs"][data["passage_id"]] = data["segmented_p"]
        else:
            lookup[data["question_id"]] = {
                "paragraphs": {data["passage_id"]: data["segmented_p"]},
                "question": data["segmented_q"],
                "question_id": data["question_id"],
                "question_type": data["question_type"],
            }
        if i % 100 == 0: print(i)

f = open("total_lookup.stat", "w")
f.write(json.dumps(lookup, ensure_ascii=False) + "\n")
f.close()
