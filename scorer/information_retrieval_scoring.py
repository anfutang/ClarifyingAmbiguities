import os
import collections
import numpy as np
import pickle
import pytrec_eval

def pointwise_mrr_score(hyp,ref,cutoff):
    gold_docid = list(ref.keys())[0]
    hyp =  hyp[:cutoff]
    if gold_docid in hyp:
        return 1.0 / (hyp.index(gold_docid)+1)
    else:
        return 0.0

def mrr_score(hyps,refs):
    # imitate pytrec_eval style
    assert len(hyps) == len(refs), "prediction and gold relevance do not have the same length."

    results = {}
    for ix, (hyp, ref) in enumerate(zip(hyps,refs)):
        results[str(ix)] = {}
        pred = [doc[0] for doc in hyp]
        # print(hyp)
        # print(pred)
        for cutoff in [3,5,10,20]:
            results[str(ix)][f'mrr_cut_{cutoff}'] = pointwise_mrr_score(pred,ref,cutoff)

    return results

def ndcg_score(hyps,refs):
    assert len(hyps) == len(refs), "prediction and gold relevance do not have the same length."

    pred, gold = {}, {}
    for ix, (hyp, ref) in enumerate(zip(hyps,refs)):
        if not ref:
            continue
        pred[str(ix)] = {doc[0]:doc[-1] for doc in hyp}
        # pred[str(ix)] = {doc[0]:1./(i+1) for i, doc in enumerate(hyp)}
        gold[str(ix)] = {docid:rel for docid,rel in ref.items() if rel > 0}
        # add this only for trec_dl_hard
        # for k, v in gold[str(ix)].items():
        #     if v == 4:
        #         gold[str(ix)][k] = 3

    evaluator = pytrec_eval.RelevanceEvaluator(gold, {'ndcg_cut.3','ndcg_cut.5','ndcg_cut.10','ndcg_cut.20'})
    results = evaluator.evaluate(pred)
    
    return results