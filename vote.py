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

def entity(s1, s2, s3):
    if s1 == "。": s1 = s2
    if s2 == "。": s2 = s3
    if s3 == "。": s3 = s2
    if s3 == "。": return [(0, s1)]
    #if s1 in s2 or s1 in s3: return [(2, s1)]
    #if s2 in s1 or s2 in s3: return [(2, s2)]
    #if s3 in s1 or s3 in s2: return [(2, s3)]
    return [(score(s1, s2) + score(s1, s3), s1),
            (score(s2, s1) + score(s2, s3), s2),
            (score(s3, s1) + score(s3, s2), s3),
            ]

def des_yes(s1, s2, s3):
    if s1 == s3 or s2 == s1: return [(2, s1)]
    if s2 == s3: return [(2, s2)]
    return [(score(s1, s2) + score(s1, s3), s1),
     (score(s2, s1) + score(s2, s3), s2),
     (score(s3, s1) + score(s3, s2), s3),
    ]    

def get_answer(s1, s2, s3, tp):
    """
    s1: a list of R-net answers
    s2: 09 BIDAF answer
    s3: 07 BIDAF answer
    """
    s2 = replace_invisible(s2)
    s3 = replace_invisible(s3)
    if len(s2) == 0:
        s2 = s3
    if len(s3) == 0: s3 = s2
    if len(s3) == 0: print("======>fuck+++")
    res = []
    for s in s1:
        s = replace_invisible(s)
        if s is None or len(s) == 0: continue
        if tp == "ENTITY": res.extend(entity(s, s2, s3))
        else: res.extend(des_yes(s, s2, s3))
    
    if len(res) == 0: return s2

    res.sort(key=lambda x: x[0], reverse=True)
    #for r in res:
    #    print(r)
    #print()
    return res[0][1]
    
    mx = 0
    s1 = replace_invisible(s1)
    s2 = replace_invisible(s2)
    s3 = replace_invisible(s3)
    if s1 is None or len(s1) == 0: return s2
    if len(s2) == 0: return s1
    if len(s3) == 0: return s1
    #if s2 == s3: return s2

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
    

w = open("pypy3_10result_tri_vote.json", "w")
ids = []
import random
for f in ["10search_result.json", "10zhidao_result.json"]:
    with open(f) as r:
        for i, line in enumerate(r):
            d = json.loads(line)
            #print("".join(d["question"]), d["question_type"])
            del d["question"]
            d["answers"] = [get_answer(d["pure_ans"], lookup09[d["question_id"]], lookup07[d["question_id"]], d["question_type"])]
            #if d["question_id"] == 237664: print(d["answers"]); break
            del d["pure_ans"]
            w.write(json.dumps(d, ensure_ascii=False) + "\n")
            ids.append(d["question_id"])
            #if i == 101: break
            if i % 100 == 0: print(i)

for k, v in _lookup09.items():
    if k not in ids:
        if random.randint(0, 1) == 0:
            w.write(json.dumps(v, ensure_ascii=False) + "\n") 
        else:
            w.write(json.dumps(_lookup07[k], ensure_ascii=False) + "\n")
w.close()
