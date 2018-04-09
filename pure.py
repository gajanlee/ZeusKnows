import json

def notStopSym(char):
    return char not in ["。", "！", "？", "："]

def Pure(passage, spans):
    s, e = spans
    answer = passage[s:e]
    if answer == ["。"] or answer[0] == "。":
        while s > 0 and notStopSym(passage[s]):
            s -= 1
    answer = passage[s:e]
    if notStopSym(answer[-1]):
        while e < len(passage) and notStopSym(passage[e]):
            e += 1
    answer = passage[s:e]
    return answer
            
#lookup = json.load(open("total_lookup.stat"))

with open("match/search_total.stat") as fp:
    lookup = [json.loads(line) for line in fp]



answers = {}
with open("res/08search_res.stat") as fp:
    for i, line in enumerate(fp):
        data = json.loads(line)
        if data["question_id"] not in answers:
            answers[data["question_id"]] = {
                "question_id": data["question_id"],
                "question_type": lookup[data["passage_id"]]["question_type"],
                "answer_spans": [data["spans"]],
                "passages": [lookup[data["passage_id"]]["segmented_p"]],
                "question": lookup[data["passage_id"]]["segmented_q"],
                "yesno_answers": [],
            }
        else:
            answers[data["question_id"]]["answer_spans"].append(data["spans"])
            answers[data["question_id"]]["passages"].append(lookup[data["passage_id"]]["segmented_p"])
        print(i)
with open("search_res.stat", "w") as w:
    for o in answers.values():
        w.write(json.dumps(o, ensure_ascii=False) + "\n")


