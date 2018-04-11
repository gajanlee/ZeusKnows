from collections import Counter
#import jieba
import json
def StopSym(char):
    return char in ["。", "！", "？", "..."]

def first_start(seq, num):
    for i, n in enumerate(seq):
        if n > num:
            a = seq[i-1] +1 if seq[i-1] != 0 else 0
            return a
            
            return seq[i-1]+1 if seq[i-1] != 0 else 0

def first_end(seq, num):
    for i, n in enumerate(seq):
        
        if n >= num: return n+1

def seg(passage, judge=StopSym):
    indexes = [0]
    for i, char in enumerate(passage):
        if judge(char):
            indexes.append(i)
    indexes.append(len(passage))
    return indexes

def Pure(passage, spans):
    if spans[0] == spans[1]: return None
    s = seg(passage)
    if "、" in passage:
        dunhao = seg(passage, lambda x: x == "、")
        return passage[first_start(s, dunhao[1]): first_end(s, dunhao[-2])]
    else:
        return passage[first_start(s, spans[0]): first_end(s, spans[1])]

"""lookup = {}
with open("../../09result.json") as r:
    for line in r:
        d = json.loads(line)
        lookup[d["question_id"]] = list(jieba.cut(d["answers"][0]))
"""
def replace_invisible(x):
    return x.replace(" ", "").replace("\n", "").replace("&nbsp;", "").replace("　", "")

def BestChoice(id, answers, tp):
    anss = []
    for ans in answers:
        if ans is None or len(ans) == 0: continue
        anss.append(replace_invisible("".join(ans)))
    return anss

    mx = (-1, None)
    for ans in answers:
        
        if ans is None or len(ans) == 0: continue
        
        # score = sum((Counter(lookup[id]) & Counter(ans)).values()) / len(ans)
        score = sum((Counter("".join(lookup[id])) & Counter("".join(ans))).values())

        if score > mx[0]:
            mx = (score, ans)
    if mx[1] == None: print("=====>", id); return lookup[id]
    return "".join(mx[1])

ans = {}
w = open("11search_result.json", "w")
with open("search_res.stat") as r:
    import json

    for i, line in enumerate(r):
        d = json.loads(line)
        
        for passage, spans in zip(d["passages"], d["answer_spans"]):
            d["pure_ans"] = [Pure(passage, spans) for passage, spans in zip(d["passages"], d["answer_spans"])]
        d["pure_ans"] = BestChoice(d["question_id"], d["pure_ans"], d["question_type"])
        del d["passages"]
        del d["answer_spans"]
        w.write(json.dumps(d, ensure_ascii=False) + "\n")
        if i % 100 == 0: print(i)


w.close()
