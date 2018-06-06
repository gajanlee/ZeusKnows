from R_net.rouge_metric.rouge import Rouge
import json

if __name__ == "__main__":
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

    idf = json.load(open("idf.stat"))
    article_count = 1294232
    #idf = json.loads("idf.stat")
def replace_invisible(x):
    return x.replace(" ", "").replace("\n", "").replace("&nbsp;", "").replace("　", "")

# 字符级别的分数判定
rouge = Rouge()
def score(s1, s2):
    if not s1 or not s2:
        return 0 
    return rouge.calc_score([s1], [s2])

def entity(s1, s2, s3):
    if s1 == "。": s1 = s2
    if s2 == "。": s2 = s3
    if s3 == "。": s3 = s2
    if s3 == "。": return [(2, s1)]
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


def del_juhao(s_list):
    res = []
    for s in s_list:
        if s == "。": continue
        for i, _s in enumerate(s):
            if _s != "。": s = s[i:]; break;
        res.append(s)
    return res

def ensemble_answer(question, docs, *ans):
    """
    ans, a list of Answer
    """
    #s1 = del_juhao(s1)
    print(ans)
    res = []
    for i, si in enumerate(ans):
        scr = sum([score(si["answers"], sj["answers"]) for j, sj in enumerate(ans) if i != j])
        res.append((scr + score(si["answers"], question), si["answers"], docs[si["passage_id"]].passage))
    
    _res = list(filter(lambda x: x[1] != "。", res))
    if not _res:
        return "No Answer"
    _res.sort(key=lambda x: x[0], reverse=True)
    return _res

if __name__ == "__main__":
    w = open("11result_tri_vote_2.json", "w")
    ids = []
    import random
    for f in ["11search_result.json", "11zhidao_result.json"]:
        with open(f) as r:
            for i, line in enumerate(r):
                d = json.loads(line)
                #print(d["question_id"], "".join(d["question"]), d["question_type"])
                #del d["question"]
                d["answers"] = [ensemble_answer("".join(d["question"]), *[d["pure_ans"], lookup09[d["question_id"]], lookup07[d["question_id"]]])]
                del d["question"]
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
