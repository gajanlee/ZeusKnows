from rouge_metric.rouge import Rouge

import logging
logger = logging.getLogger("synethsis answer")
logging.basicConfig(level = logging.DEBUG, 
                    format = "%(asctime)s : %(levelname)s  %(message)s",
                    datefmt = "%Y-%m-%d %A %H:%M:%S")
from functools import wraps
def logging_util(func):
    @wraps(func)
    def with_logging(*args, **kwargs):
        logger.info("Starting %s, args: %s" % (func.__name__, args[1:]))
        return func(*args, **kwargs)
    return with_logging


def StopSym(char):
    return char in ["。", "！", "？", "..."]

def first_start(seq, num):
    for i, n in enumerate(seq):
        if n > num:
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

def expand_answer(passage, spans):
    if spans[0] == spans[1]: return None
    s = seg(passage)
    if "、" in passage:
        dunhao = seg(passage, lambda x: x == "、")
        return "".join(passage[first_start(s, dunhao[1]): first_end(s, dunhao[-2])])
    else:
        return "".join(passage[first_start(s, spans[0]): first_end(s, spans[1])])

def replace_invisible(x):
    """
    replace x's invisible character.
    """
    return x.replace(" ", "").replace("\n", "").replace("&nbsp;", "").replace("　", "")

# 字符级别的分数判定
rouge = Rouge()
def score(s1, s2):
    """
    s1, s2: strings.
    s1 is candidate string, s2 is reference string.
    """
    return rouge.calc_score([s1], [s2])

def entity(answer):
    """
    answer: string, selected answer
    return:
        [a list of entity answers]
    """
    import jieba
    res = []
    for e in list(jieba.cut(answer)):
        if idf(e) >= 3.0:
            res.append(e)
    return res

@logging_util
def idf(token):
    import json, math
    idf_dict = json.loads("./idf.dict")
    article_count = 1294233
    def inner():
        return math.log(article_count / idf_dict.get(token, 1))
    return inner()



