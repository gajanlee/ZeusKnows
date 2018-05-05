# -*- coding: utf-8 -*-
"""
    docs.py is used to machine computer comprehension,
    when user upload a question, we need to get related documents.
"""
from __future__ import print_function
import requests
from bs4 import BeautifulSoup
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import threading
import Queue

docs = Queue.Queue()    # threading safety
class Document:
    def __init__(self, *data):
        self.link, self.title, self.passage = data

class ZhidaoThreadDownload(threading.Thread):
    def __init__(self, url):
        super(ZhidaoThreadDownload,self).__init__()
        self.url = url

    def download_docs(url):
        pass

    def run(self, keyword, n=5):
        # the details is encoded by utf-8
        detail_soup = BeautifulSoup(unicode(requests.get(self.url).content, "gb2312", "ignore"), "html5lib")   # 忽略非gb2312的编码
        title = detail_soup.find(attrs={"class":"ask-title"}).text
        best_answer = detail_soup.find(attrs={"class": "best-text"})
        if best_answer is None: return    # 没有最佳答案
        docs.put(Document(self.url, title, best_answer.text))

def download_docs(url):
    docs.queue.clear()
    r = requests.get(url)
    lst_soup = BeautifulSoup(unicode(r.content, "gb2312", "ignore"), "html5lib")
    items = lst_soup.find_all("dl", attrs={"class": "dl"})[:7]
    threads = []
    for item in items:
        t = ZhidaoThreadDownload(item.dt.a["href"])
        t.start()
        #t.join()
        threads.append(t)
    for t in threads:
        t.join()

def get_docs(sentence, n=5):
    docs.queue.clear()
    download_docs('https://zhidao.baidu.com/search?lm=0&rn=10&pn=0&fr=search&ie=gbk&word=' + sentence)

    r = requests.get('https://zhidao.baidu.com/search?lm=0&rn=10&pn=0&fr=search&ie=gbk&word=' + sentence)
    lst_soup = BeautifulSoup(unicode(r.content, "gb2312", "ignore"), "html5lib")
    items = lst_soup.find_all("dl", attrs={"class": "dl"})[:int(n*1.4)]
    threads = []
    for item in items:
        t = ZhidaoThreadDownload(item.dt.a["href"])
        t.start()
        #t.join()
        threads.append(t)
    for t in threads:
        t.join()

if __name__ == "__main__":
    getDocs("浦发银行电话号码")
    while not docs.empty():
        print(docs.get().title)