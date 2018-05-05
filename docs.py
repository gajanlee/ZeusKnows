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

class Downloader(object):
    def __init__(self, keyword):
        self.docs = Queue.Queue()
        self.url = self.synthesis_url(keyword)        
    
    def start(self):
        Docs = []
        self.start_downloading()
        while not self.docs.empty():
            Docs.append(self.docs.get())
        return Docs

    def synthesis_url(self, keyword):
        raise NotImplementedError

    def start_downloading(self):
        raise NotImplementedError

class Zhidao_Downloader(Downloader):
    def __init__(self, keyword):
        super(Zhidao_Downloader, self).__init__(keyword)
    
    def synthesis_url(self, keyword):
        return 'https://zhidao.baidu.com/search?lm=0&rn=10&pn=0&fr=search&ie=gbk&word=' + keyword
    
    def start_downloading(self):
        list_soup = get_html_content_soup(self.url)
        threads = []
        for item in list_soup.find_all("dl", attrs={"class": "dl"})[:7]:
            t = threading.Thread(target=self.download_one, args=(item.dt.a["href"], self.docs))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    @staticmethod
    def download_one(url, docs):
        item_soup = get_html_content_soup(url)
        best_answer = item_soup.find(attrs={"class": "best-text"})
        if best_answer is None: return
        docs.put(Document(url, item_soup.find(attrs={"class":"ask-title"}).text, best_answer.text))

def get_html_content_soup(url):
    return BeautifulSoup(unicode(requests.get(url).content, "gb2312", "ignore"), "html5lib")   # 忽略非gb2312的编码(ignore)


"""
    Exterior interfaces
"""
def get_docs(keyword):
    return Zhidao_Downloader("浦发银行电话号码").start()

if __name__ == "__main__":
    for doc in Zhidao_Downloader("浦发银行电话号码").start():
        print(doc.title)