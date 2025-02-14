import os
import json
import pickle
from pygaggle.rerank.base import Query, Text, hits_to_texts
from pygaggle.rerank.transformer import MonoT5
from pyserini.search.lucene import LuceneSearcher

if __name__ == "__main__":
    args = get_args()

    logging_dir, output_dir = build_dst_folder(args,True)
    logging_filename = os.path.join(logging_dir,"log.log")
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%m/%d/%Y %H:%S',level=logging.INFO,filename=logging_filename,filemode='w')

    logging.info(show_job_infos(args,True))    

    searcher = LuceneSearcher.from_prebuilt_index(args.prebuilt_index_name)
    logging.info("Retriever loaded: BM25.")
    reranker =  MonoT5()
    logging.info("Reranker loaded: MonoT5.")

    qs = json.load(open(os.path.join(args.data_dir,"ir",f"{args.dataset_name}.json")))["query"]

    if args.stage == "retrieve":
        retrieve_result, rerank_result = [], []
        for q in qs:
            hits = searcher.search(q,k=self.k)
            texts = hits_to_texts(hits)
            retrieve_result.append([[r.metadata["docid"],r.score] for r in texts])
        
        res = {"retrieve":retrieve_result}
    else:
        retrieve_result, rerank_result = [], []
        for q in qs:
            hits = searcher.search(q,k=self.k)
            texts = hits_to_texts(hits)
            reranked = reranker.rerank(Query(rq),texts)
            retrieve_result.append([[r.metadata["docid"],r.score] for r in texts])
            rerank_result.append([[r.metadata["docid"],r.score] for r in reranked])
        
        res = {"retrieve":retrieve_result,"rerank":rerank_result}

    with open(os.path.join(output_dir,"retrieval_result.pkl"),"wb") as f:
        pickle.dump(res,f)