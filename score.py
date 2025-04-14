import os
import time
import datetime
import logging
import pickle
from opt import get_args
from scorer.scorer import Scorer
from utils.utility import build_dst_folder, validate_arguments

def make_dir_if_not_exist(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

if __name__ == "__main__":
    args = get_args()
    validate_arguments(args)

    if args.ir_no_clarification:
        logging_dir = os.path.join(args.logging_dir,args.dataset_name,"ir_no_clarification")
        make_dir_if_not_exist(logging_dir)

        output_dir = os.path.join(args.output_dir,args.dataset_name,"ir_no_clarification")
        make_dir_if_not_exist(output_dir)
    else:
        logging_dir, output_dir = build_dst_folder(args)

    logging_filename = os.path.join(logging_dir,f"{args.score_type}_score.log")
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%m/%d/%Y %H:%M',level=logging.INFO,filename=logging_filename,filemode='w')

    logging.info("*Start time:"+str(datetime.datetime.now()))
    if args.score_type == "cq":
        logging.info("*Score type: CQ.")
    else:
        logging.info("*Score type: IR.")
        logging.info(f"*Score stage: {args.score_stage}")

    start_time = time.time()

    scores = Scorer(args).score()
    
    with open(os.path.join(output_dir,f"{args.score_type}_result.pkl"),"wb") as f:
        pickle.dump(scores,f,pickle.HIGHEST_PROTOCOL)

    logging.info(f"Scores saved under {output_dir}.")

    end_time = time.time()
    logging.info(f"Finished: {end_time-start_time} s.")
