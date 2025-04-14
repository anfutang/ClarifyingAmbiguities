import os
import json
import pickle
import numpy as np
from scipy.stats import ttest_ind, wilcoxon
from scorer.information_retrieval_scoring import *
from opt import get_args

# prompt_types = ["zero-shot","AT-zero-shot","CoT-zero-shot","AT-CoT-zero-shot",
#                 "few-shot","AT-few-shot","CoT-few-shot","AT-CoT-few-shot"]
user_simulation_modes = ["select","respond","select+respond"]

prompt_types = ["few-shot","AT-few-shot","CoT-few-shot","AT-CoT-few-shot"]

combinations = [('AT-few-shot', 'few-shot'), ('CoT-few-shot', 'few-shot'), ('CoT-few-shot', 'AT-few-shot'), 
                ('AT-CoT-few-shot', 'few-shot'), ('AT-CoT-few-shot', 'AT-few-shot'), ('AT-CoT-few-shot', 'CoT-few-shot')]

def interpret_cq_scores(scores):
    has_empty = False
    for score in scores:
        if not score:
            has_empty = True
    assert not has_empty, "empty score for a given example."

    # computes the maximum similarity between a generated CQ and all reference CQs as the score fot the generated CQ.
    aves = []
    for score in scores:
        aves.append(np.array(score).max(axis=0).mean())
    return aves

def statistical_test(scores,alpha):
    output = []
    for pt1, pt2 in combinations:
        p = ttest_ind(scores[pt1],scores[pt2],alternative="greater").pvalue
        if p < alpha:
            output.append(f"({pt1},{pt2}): {p}")
    return '\n'.join(output)

def build_qrels(qrels):
    flattened_qrels = []
    for qrel in qrels:
        flattened_qrels += qrel
    return flattened_qrels

def flatten_preds(preds,ls):
    flatten_preds = []
    for pred, l in zip(preds,ls):
        flatten_preds += [pred] * l
    return flatten_preds
 
if __name__ == "__main__":
    args = get_args()

    assert args.score_type in ["cq","ir"], "ONLY two scoring types supported: 'cq' or 'ir'."

    if args.score_type == "cq":
        base_dir = os.path.join(args.output_dir,args.dataset_name,f"noise_type_{args.noise_type}","turn_1","score","select+respond")

        scores = {}
        output = [f"Dataset: {args.dataset_name}.",f"===== Bert Score (Deberta) =====\n"]

        for pt in prompt_types:
            dst_dir = os.path.join(base_dir,pt)
            if not os.path.exists(dst_dir):
                output.append(f"missing output file for prompt type {pt}.\n")
            else:
                scores[pt] = interpret_cq_scores(pickle.load(open(os.path.join(dst_dir,"cq_result.pkl"),"rb")))

        output.append("===== ave ± std =====")
        for pt, score in scores.items():
            output.append(f"{pt}: {np.mean(score)} ± {np.std(score)}")
        
        output.append('\n===== statistical test (one-side t-test \'greater\') =====')
        
        output.append(statistical_test(scores,args.alpha))

        with open(os.path.join(args.score_dir,f"{args.score_type}_{args.dataset_name}.txt"),'w') as f:
            f.write('\n'.join(output))

    if args.score_type == "ir":
        if not args.ir_no_clarification:
            base_dir = os.path.join(args.output_dir,args.dataset_name,f"noise_type_{args.noise_type}",f"turn_{args.turn_id}","score")
            gold_rels = build_qrels(json.load(open(os.path.join(args.data_dir,f"{args.dataset_name}.json")))["relevance"])

            all_scores = {}

            for score_stage in ["retrieve","rerank"]:
                all_scores[score_stage] = {}
                for usm in user_simulation_modes:
                    all_scores[score_stage][usm] = {}
                    for pt in prompt_types:
                        preds = pickle.load(open(os.path.join(base_dir,usm,pt,f"ir_result.pkl"),"rb"))[score_stage]
                        # res = [[s[0] for s in r] for r in res]
                        if args.ir_eval_metric == "ndcg":
                            results = ndcg_score(preds,gold_rels)
                        elif args.ir_eval_metric == "mrr":
                            results = mrr_score(preds,gold_rels)
                        all_scores[score_stage][usm][pt] = results
            
            tmp_dir_names = [args.dataset_name,f"noise_type_{args.noise_type}"]
            score_dir = args.score_dir
            for tdn in tmp_dir_names:
                score_dir = os.path.join(score_dir,tdn)
                if not os.path.exists(score_dir):
                    os.makedirs(score_dir)
            
            with open(os.path.join(score_dir,f"{args.score_type}_{args.ir_eval_metric}_turn_{args.turn_id}.pkl"),'wb') as f:
                pickle.dump(all_scores,f,pickle.HIGHEST_PROTOCOL)
        else:
            base_dir = os.path.join(args.output_dir,args.dataset_name,"ir_no_clarification")
            source_data = json.load(open(os.path.join(args.data_dir,f"{args.dataset_name}.json")))
            ls = list(map(len,source_data["user_intention"]))
            gold_rels = build_qrels(source_data["relevance"])
            pred_data = pickle.load(open(os.path.join(base_dir,f"ir_result.pkl"),"rb"))

            all_scores = {}
            for score_stage in ["retrieve","rerank"]:
                preds = flatten_preds(pred_data[score_stage],ls)
                if args.ir_eval_metric == "ndcg":
                    results = ndcg_score(preds,gold_rels)
                elif args.ir_eval_metric == "mrr":
                    results = mrr_score(preds,gold_rels)
                all_scores[score_stage] = results

            tmp_dir_names = [args.dataset_name,"no_clarification"]
            score_dir = args.score_dir
            for tdn in tmp_dir_names:
                score_dir = os.path.join(score_dir,tdn)
                if not os.path.exists(score_dir):
                    os.makedirs(score_dir)

            with open(os.path.join(score_dir,f"{args.score_type}_{args.ir_eval_metric}_no_clarification.pkl"),'wb') as f:
                pickle.dump(all_scores,f,pickle.HIGHEST_PROTOCOL)


    