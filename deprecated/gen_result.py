import json

lookup = json.load(open("total_lookup.stat"))

writer = open("result.json", "w")
#writerh = open("result_h.json", "w")
#print(lookup["301121"]["paragraphs"]["58075"])
output = {}
output_h = {}
#for tp in ["yes_no", "entity", "description"]:
for _ in range(1):
#    with open("../res/{}_test_res.stat".format(tp)) as fp:
    with open("") as fp:
        for u, line in enumerate(fp, 1):
            data = json.loads(line)

            s1, s2 = data["spans"]
            if s1 == s2: continue
            if "".join(lookup[str(data["question_id"])]["paragraphs"][str(data["passage_id"])][s1: s2]) in ["。", "."]: continue
            if lookup[str(data["question_id"])]["paragraphs"][str(data["passage_id"])][s1] == "。": s1 += 1
            if data["question_id"] not in output:
                output[data["question_id"]] = {
                    "question_id": data["question_id"],
                    "question_type": lookup[str(data["question_id"])]["question_type"],
                    "answers": ["".join(lookup[str(data["question_id"])]["paragraphs"][str(data["passage_id"])][s1: s2]) ],
                    "yesno_answers": [],
                    "entity_answers": [[]],
                }
            else:
                output[data["question_id"]]["answers"].append(
                    "".join(lookup[str(data["question_id"])]["paragraphs"][str(data["passage_id"])][s1: s2])
                )
            if u % 100 == 0: print(u)

for o in output.values():
    writer.write(json.dumps(o, ensure_ascii=False) + "\n")
writer.close()
