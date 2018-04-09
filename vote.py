from rouge_metric.rouge import Rouge
import json

lookup09 = {}
_lookup09 = {}
with open("../../09result.json") as r:
    for line in r:
        #print(line)
        d = json.loads(line)
        lookup09[d["question_id"]] = d["answers"][0]
        _lookup09[d["question_id"]] = d

lookup07 = {}
_lookup07 = {}
with open("../../07result.json") as r:
    for line in r:
        d = json.loads(line)
        lookup07[d["question_id"]] = d["answers"][0]
        _lookup07[d["question_id"]] = d

def replace_invisible(x):
    return x.replace(" ", "").replace("\n", "").replace("&nbsp;", "")

# 字符级别的分数判定
rouge = Rouge()
def score(s1, s2):
    return rouge.calc_score([s1], [s2])

def get_answer(s1, s2, s3):
    
    mx = 0
    s1 = replace_invisible(s1)
    s2 = replace_invisible(s2)
    s3 = replace_invisible(s3)
    if s1 is None or len(s1) == 0: return s2
    if len(s2) == 0: return s1
    if len(s3) == 0: return s1
    if s2 == s3: return s2

    score1 = score(s1, s2) + score(s1, s3); mx = score1
    score2 = score(s2, s1) + score(s2, s3); 
    if score2 > mx: mx = score2
    score3 = score(s3, s1) + score(s3, s2); 
    if score3 > mx: mx = score3
    if False:
        print(mx)
        print(score1, s1)
        print(score2, s2)
        print(score3, s3, "\n")

    if mx == score1: return s1
    if mx == score2: return s2
    return s3
    

w = open("09result_tri_vote.json", "w")
ids = []
import random
for f in ["search_result.json", "zhidao_result.json"]:
    with open(f) as r:
        for i, line in enumerate(r):
            d = json.loads(line)
            #print(d["question"])
            del d["question"]
            d["answers"] = [get_answer("".join(d["pure_ans"]), lookup09[d["question_id"]], lookup07[d["question_id"]])]
            del d["pure_ans"]
            w.write(json.dumps(d, ensure_ascii=False) + "\n")
            ids.append(d["question_id"])
            #if i == 100: break
            if i % 100 == 0: print(i)

for k, v in _lookup09.items():
    if k not in ids:
        if random.randint(0, 1) == 0:
            w.write(json.dumps(v, ensure_ascii=False) + "\n") 
        else:
            w.write(json.dumps(_lookup07[k], ensure_ascii=False) + "\n")
w.close()
