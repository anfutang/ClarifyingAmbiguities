import os
import sys
import logging
import json
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

dataset2indexes = {"trecweb1314":"INDEX_TO_TRECWEB1314",
                   "qulac":"INDEX_TO_TRECWEB0312",
                   "trec_dl_hard":"msmarco-passage/trec-dl-hard"}

logger = logging.getLogger(__name__)

def get_content(s):
    return s[s.index('"contents"'):].replace('"contents"','').strip().strip(':').strip('}').strip('\n').strip().strip('"')

class Scorer:
    def __init__(self,args):
        self.score_type = args.score_type
        assert self.score_type in ["cq","ir"], "ONLY two scoring types allowed: 'cq' or 'ir'."

        self.score_stage = args.score_stage
        assert self.score_stage in ["retrieve","rerank","retrieve+rerank","-"], "scoring stages allowed: 'retrieve', 'rerank', 'retrieve+rerank', '-' for IR."
        self.k = args.k

        self.dataset_name = args.dataset_name
        self.turn_id = args.turn_id
        self.stage = args.stage
        self.user_simulation_mode = args.user_simulation_mode
        self.prompt_type = args.prompt_type
        self.noise_type = args.noise_type
        self.ir_no_clarification = args.ir_no_clarification
        self.dry_run = args.dry_run
        self.dry_run_number_of_examples = args.dry_run_number_of_examples
        
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','data'))
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','output'))
    
    def score(self):
        if self.score_type == "cq":
            from .clarification_question_scoring import cq_score

            refs = json.load(open(os.path.join(self.data_dir,f"{self.dataset_name}.json")))["reference_clarification_questions"]
            # cq_source_data = json.load(open(os.path.join(self.output_dir,self.dataset_name,f"noise_type_{self.noise_type}",f"turn_1","generation","select+respond",self.prompt_type,"output.json")))["output"]
            cq_source_data = json.load(open(os.path.join(self.output_dir,self.dataset_name,f"noise_type_{self.noise_type}",f"turn_1","generation","respond",self.prompt_type,"output.json")))["output"]
            logger.info("data loaded.")

            # hyps = [doc["processed"]["clarification_questions"] for doc in cq_source_data]
            hyps = [[doc["processed"]["clarification_question"]] for doc in cq_source_data]
            assert len(hyps) == len(refs), "unequal number of reference questions and generated clarification questions."
            
            has_empty = False
            for ref in refs:
                if not ref or min(map(len,ref)) == 0:
                    has_empty = True
            assert not has_empty, "reference questions contain empty sequences or empty string in a certain sequence."

            has_empty = False
            for hyp in hyps:
                if not hyp or min(map(len,hyp)) == 0:
                    has_empty = True
            assert not has_empty, "generated clarification questions contain empty sequences or empty string in a certain sequence."

            if self.dry_run:
                refs = refs[:self.dry_run_number_of_examples]
                hyps = hyps[:self.dry_run_number_of_examples]
                logger.info(f"DRY RUN MODE: test on first {len(refs)} examples.")
            else:
                logger.info(f"\t# examples: {len(refs)}.")
            
            return cq_score(refs,hyps)
        
        if self.score_type == "ir":
            if not self.ir_no_clarification:
                summary = json.load(open(os.path.join(self.output_dir,self.dataset_name,f"noise_type_{self.noise_type}",f"turn_{self.turn_id}","summary.json")))[self.user_simulation_mode][self.prompt_type]
                rqs = summary["reformulated_query"]
                logger.info("data loaded; w/ clarification.")
            else:
                if self.dataset_name != "trecweb1314":
                    rqs = json.load(open(os.path.join(self.data_dir,f"{self.dataset_name}.json")))["query"]
                    logger.info("data loaded: original query; w/o clarification.")
                else:
                    rqs = json.load(open(os.path.join(self.data_dir,f"{self.dataset_name}_sentence_queries.json")))
                    logger.info("data loaded: transformed sentence query; w/o clarification.")

            if self.dry_run:
                rqs = rqs[:self.dry_run_number_of_examples]
                logger.info(f"DRY RUN MODE: test on first {len(rqs)} examples.")
            else:
                logger.info(f"\t# examples: {len(rqs)}.")

            if self.score_stage == "retrieve":
                searcher = self.load_bm25_researcher()

                retrieve_result = []
                for rq in rqs:
                    hits = searcher.search(rq,k=self.k)
                    passages = [[hit.docid,get_content(searcher.doc(hit.docid).raw()),hit.score] for hit in hits]
                    retrieve_result.append(passages)
                
                res = {self.score_stage:retrieve_result}

            if self.score_stage == "rerank":
                import torch
                from pygaggle.rerank.base import Query, Text
                from pygaggle.rerank.transformer import MonoT5

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                reranker = MonoT5()
                reranker.model.to(device)
                logging.info(f"Reranker loaded: MonoT5 on device {device}.")
                
                try:
                    if not self.ir_no_clarification:
                        res = pickle.load(open(os.path.join(self.output_dir,self.dataset_name,f"noise_type_{self.noise_type}",f"turn_{self.turn_id}",self.stage,self.user_simulation_mode,self.prompt_type,"ir_result.pkl"),"rb"))
                    else:
                        res = pickle.load(open(os.path.join(self.output_dir,self.dataset_name,"ir_no_clarification","ir_result.pkl"),"rb"))
                except:
                    logging.error("No retrieval result: you need to retrieve first.")
                
                rerank_result = []
                for q, p in zip(rqs,res["retrieve"]):
                    q = Query(q)
                    p = [Text(passage[1], {'docid':passage[0]}, 0) for passage in p]
                    reranked = reranker.rerank(q,p)
                    rerank_result.append([[r.metadata["docid"],r.score] for r in reranked])

                res[self.score_stage] = rerank_result

            if self.score_stage == "retrieve+rerank":
                import torch
                from pygaggle.rerank.base import Query, Text, hits_to_texts
                from pygaggle.rerank.transformer import MonoT5

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                searcher = self.load_bm25_researcher()
                reranker =  MonoT5()
                reranker.model.to(device)
                logging.info(f"Reranker loaded: MonoT5 on device {device}.")

                retrieve_result, rerank_result = [], []
                for rq in rqs:
                    hits = searcher.search(rq,k=self.k)
                    texts = hits_to_texts(hits)
                    reranked = reranker.rerank(Query(rq),texts)
                    retrieve_result.append([[r.metadata["docid"],r.score] for r in texts])
                    rerank_result.append([[r.metadata["docid"],r.score] for r in reranked])
                
                res = {"retrieve":retrieve_result,"rerank":rerank_result}

            return res
    
    def load_bm25_researcher(self):
        from pyserini.search.lucene import LuceneSearcher

        index_name = dataset2indexes[self.dataset_name]

        if self.dataset_name in ["trec_dl_hard"]:
            searcher = LuceneSearcher.from_prebuilt_index(index_name)
        else:
            searcher = LuceneSearcher(index_name)

        logger.info(f"BM25 researcher loaded; indexes from: {index_name}.")
        return searcher