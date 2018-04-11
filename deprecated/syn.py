import json

from rouge_metric.rouge import Rouge

w = open("08result.json", "w")
ids = []
for tp in ["search_result.json", "zhidao_result.json"]:
    with open(tp) as f:
        for line in f:
            d = json.loads(line)
            del d["question"]
            d["answers"] = ["".join(d["pure_ans"])]
            del d["pure_ans"]
            w.write(json.dumps(d, ensure_ascii=False) + "\n")
            ids.append(d["question_id"])
print(len(ids))
print("asdfafsafds")
lookup09 = {}
with open("../../v2/test.predicted.json") as r:
    for line in r:
        d = json.loads(line)
        lookup[d["question_id"]] = d
print(len(lookup))
count = 1
for k, v in lookup.items():
    if k not in ids:
        w.write(json.dumps(v, ensure_ascii=False) + "\n")
        count += 1
w.close()
print(count)
