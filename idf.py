import json
f = open("large_pre.dat")
idf = {}
# article_count is 1294233
for i, line in enumerate(f, 1):
    words = line[:-1].split(" ")
    ex = {}
    for word in words:
        ex[word] = True
    for word in ex.keys():
        idf[word] = idf.get(word, 0) + 1
    if i % 10 == 0: print(i)
#
with open("idf.stat", "w") as w:
    w.write(json.dumps(idf, ensure_ascii=False) + "\n")


f.close()
