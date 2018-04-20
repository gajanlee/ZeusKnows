from rouge_metric.rouge import Rouge

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
        return passage[first_start(s, dunhao[1]): first_end(s, dunhao[-2])]
    else:
        return passage[first_start(s, spans[0]): first_end(s, spans[1])]

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

import logging
logger = logging.getLogger("synethsis answer")
logging.basicConfig(level = logging.DEBUG, 
                    format = "%(asctime)s : %(levelname)s  %(message)s",
                    datefmt = "%Y-%m-%d %A %H:%M:%S")

from functools import wraps
def logging_util(func):
    @wraps(func)
    def with_logging(*args, **kwargs):
        logger.info("Starting %s" % (func.__name__))
        return func(*args, **kwargs)
    return with_logging

