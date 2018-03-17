from tqdm import tqdm 
import json

def average():
    n = avg = ls = 0
    while True:
        l = yield avg
        n += 1
        ls += l
        avg = ls / n

m_par = average(); next(m_par)
m_ans = average(); next(m_ans)
m_word = average(); next(m_word)

class Word:
    def __init__(self, id, content):
        self.count = 1
        self.content = content
        self.id = id
        self.length = len(content)

    def inc(self):
        self.count += 1

    def __str__(self):
        return "<Word id=%s content=%s count=%s>" % (self.id, self.content, self.count)
    
    def __repr__(self):
        return "<Word id=%s content=%s count=%s>" % (self.id, self.content, self.count)



def vocab_reader(filepaths, start_line=0, end_line=None):
    for filepath in filepaths:
        with open(filepath) as f:
            for _ in range(start_line): next(f)
            for i, l in zip(tqdm(range(90366)), f):
                if end_line is not None and i+start_line == end_line: break
                tp = preprocess_word_list(json.loads(l))
    return tp

import re
def filter(word):
    return True if re.findall(re.compile(r'[a-zA-Z0-9]'), word) or len(word) > 8 else False
    """for ch in list(word):
 `       if ch.isalpha() or ch.isalnum():
            return True
    return False""" # bug, chinese characters are also 'alpha'

max_paragraph = max_answer = 0
mean_paragraph = mean_answer = 0
word_list = []
def preprocess_word_list(data):
    # default value, some documents no answers
    global max_paragraph, max_answer
    global mean_paragraph, mean_answer
    for doc in data["documents"]:
        para_len = 0
        for para in doc["segmented_paragraphs"]:
            temp_word_list = []
            for word in para:
                process(word)
                temp_word_list.append(WordID(word))
            para_len += len(para)
            word_list.append(temp_word_list)
        mean_paragraph = m_par.send(para_len)
        if para_len > max_paragraph: max_paragraph = para_len
    for answer in data["segmented_answers"]:
        for word in answer:
            process(word)
        mean_answer = m_ans.send(len(answer))
        if len(answer) > max_answer: max_answer = len(answer)
    return max_paragraph, mean_paragraph, max_answer, mean_answer
        

word_dict = {'<unknown>': Word(0, '<unknown>')}
char_dict = {'<unknown>': 0}

id ,char_id = 1, 1
mean_word_len, max_word_len = 0, 0
long_word_list = []
def process(word):
    global id, char_id, mean_word_len, max_word_len
    if filter(word): return
    for char in word:
        if char not in char_dict:
            char_dict[char] = char_id
            char_id += 1

    if word not in word_dict:
        word_dict[word] = Word(id, word)
        id += 1
    else:
        word_dict[word].inc()

    max_word_len = len(word) if len(word) > max_word_len else max_word_len
    #if len(word) > 8 and word not in long_word_list:
    #    long_word_list.append(word)
    mean_word_len = m_word.send(len(word))

def WordID(word):
    if word in word_dict:
        return word_dict[word].id
    else:
        return 0

def statistics(max_paragraph, mean_paragraph, max_answer, mean_answer):
    with open("stat.log", "w") as f:
        f.write("""max passage length is {}
        mean passage length is {}
        paragraph count is 90366    # lines   
        max answer length is {}
        mean answer length is {}
        vocabulary count is {}
        characterizer count is {}
        max word length is {}
        mean word length is {}
        long word list is {}
        """.format(max_paragraph, mean_paragraph, max_answer, mean_answer, len(word_dict), len(char_dict), max_word_len, mean_word_len, long_word_list))

def main():
    statistics(*vocab_reader(["../Dureader/data/preprocessed/trainset/zhidao.train.json"], end_line=None))
    with open("word.dict", "w") as f:
        for word in sort_dict(word_dict):
            f.write("%s %s %s %s\n" % (word.id, word.content, word.count, word.length))
    with open("char.dict", "w") as f:
        for char, id in char_dict.items():
            f.write("%s %s\n" % (char, id))
    with open("word_list.dict", "w") as f:
        for wlst in word_list:
            #print(wlst)
            f.write("%s\n" % ' '.join(map(str, wlst)))

def sort_dict(dict_words):
    """
   
    :param dict_words:
    :return:
    """
    values = dict_words.values()
 
    list_one = [v for v in values]
    #print(list_one)
    list_sort = sorted(list_one, key=lambda x: x.count)
    return list_sort


if __name__ == "__main__":
    main()
    
    
