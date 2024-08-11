from zsre_dataset import Dataset
import json
import ipdb

def safe_divide(a: float, b: float) -> float:
    if a == 0 or b == 0:
        return 0
    return a / b
def compute_score(path_pred, path_gold):
    pred = Dataset.load(path_pred)
    gold = Dataset.load(path_gold)
    assert len(pred.sents) == len(gold.sents)
    num_pred = 0
    num_gold = 0
    num_correct = 0

    for i in range(len(gold.sents)):
        num_pred += len(pred.sents[i].triplets)
        num_gold += len(gold.sents[i].triplets)
        for p in pred.sents[i].triplets:
            for g in gold.sents[i].triplets:
                if (p.head, p.tail, p.label) == (g.head, g.tail, g.label):
                    num_correct += 1

    precision = safe_divide(num_correct, num_pred)
    recall = safe_divide(num_correct, num_gold)

    info = dict(
        # path_pred=path_pred,
        # path_gold=path_gold,
        precision=precision,
        recall=recall,
        score=safe_divide(2 * precision * recall, precision + recall),
    )
    return info
path_pred_opt = './pred_our.jsonl'
path_pred_rel = './pred_rel.jsonl'
path_pred_tgm = './pred_tgm.jsonl'
path_gold = './test.jsonl'
# results = compute_score(path_pred_our, path_gold)
# print(json.dumps(results, indent=2))
sents_opt = Dataset.load(path_pred_opt).sents
sents_tgm = Dataset.load(path_pred_tgm).sents
sents_rel = Dataset.load(path_pred_rel).sents
gold_sents = Dataset.load(path_gold).sents
# sents_rp = Dataset.load(path_pred_rp).sents

for i in range(len(gold_sents)):
    trip_1 = sents_opt[i].triplets
    trip_2 = sents_tgm[i].triplets
    trip_3 = sents_rel[i].triplets
    trip_4 = gold_sents[i].triplets
    for o in trip_1:
        for t in trip_2:
            for r in trip_3:
                for g in trip_4:
                    if (o.head, o.tail, o.label) == (g.head, g.tail, g.label) and (o.head, o.tail, o.label) == (t.head, t.tail, t.label):
                        if (o.head, o.tail, o.label) != (r.head, r.tail, r.label) and o.label not in ['religion']:# and o.label not in ['religion','operating system']
                            print(o.tokens[o.head[0]:o.head[-1]+1])
                            print(o.tokens[o.tail[0]:o.tail[-1]+1])
                            print(o.label)
                            print('######################')
                            print(r.tokens[r.head[0]:r.head[-1] + 1])
                            print(r.tokens[r.tail[0]:r.tail[-1] + 1])
                            print(r.label)
                            print('######################')
                            print(t.tokens[t.head[0]:t.head[-1] + 1])
                            print(t.tokens[t.tail[0]:t.tail[-1] + 1])
                            print(t.label)
                            print('######################')
                            print(' '.join(o.tokens))
                            ipdb.set_trace()
                        # ipdb.set_trace()