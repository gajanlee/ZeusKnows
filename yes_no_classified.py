import logging
import fasttext
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

classifier = fasttext.supervised("./yes_no_wellformat_train.stat", "./yes_no_classified.model", label_prefix="__label__", dim=300, epoch=2, word_ngrams=1, ws=10  )

result = classifier.test("./yes_no_wellformat_dev.stat")
print(result.precision)
print(result.recall)
print(result.nexamples)
print(classifier.predict_proba(['对于 散 在 的 比较 小 的 宫颈 腺 囊肿 一般 不需 治疗 ， 只要 每年 检查 即可 ； 对于 密集 的 较 小 的 纳氏囊肿 或 比较 大 的 囊肿 ， 可 考虑 光疗 、 激光 、 微波 、 自凝 刀 等 物理治疗 ； 对于 较 大 的 突出 于 宫颈 表面 的 ， 可 考虑 电刀 切除 治疗 。']))   # 需要空格分

