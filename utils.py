from rouge_metric.rouge import Rouge

# 字符级别的分数判定
rouge = Rouge()
def score(s1, s2):
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