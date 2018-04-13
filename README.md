# ZeusKnows
A QA system based on computer reading comprehension.

## Preprocess
### 训练词向量
1. word2vec
```bash
pip install word2vec
```
2. 训练
```python
word2vec.word2vec('large_pre.dat', 'wordsVec.bin', size=300, verbose=True, sample=8, cbow=0,  threads=4, min_count=40, save_vocab="word2vec.dict")
```

### 训练Yes/No 判断程序
* [Fasttext](https://github.com/facebookresearch/fastText/)

## 生成输入数据

1. generate_net.py
```
从DuReader数据中生成成为R-net data_loader可识别的数据，添加ID。
```
2. 运行R-net，生成模型。

3. vote.py, ensemble，从多个候选答案中选择最靠谱的一个。

4. 添加yes_no 的答案，在95 服务器


## Tips:
1. word_embedding 40count 运行中
* 95: /home/libei/ljz/sougou_data/word_50.log
* aliyun: ZeusKnows/word2vec.log


## TODO:
1. add language_model(pytorch) implementation.
* 把所有的有关预处理和文章排序文件下放到另外的一个文件夹。
* 把language_model作为一个包。__init__.py可以引用vocabulary.
* 