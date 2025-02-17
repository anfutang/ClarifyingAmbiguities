import argparse
from argparse import ArgumentParser

# "type" in argparse could be a function.
def str2bool(arg):
    if isinstance(arg, bool):
       return arg
    if arg.lower() in ('true','1'):
        return True
    elif arg.lower() in ('false','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
     parser = ArgumentParser(description="prompting with ambiguity type definitions.")
     parser.add_argument("--data_dir",type=str,default="./data/")
     parser.add_argument("--output_dir",type=str,default="./output/")
     parser.add_argument("--prompt_dir",type=str,default="./prompt/")
     parser.add_argument("--logging_dir",type=str,default="./logging/")
     parser.add_argument("--score_dir",type=str,default="./score/")
     parser.add_argument("--model_name",type=str,default="llama3-8b")
     parser.add_argument("--dataset_name",type=str,required=True)
     parser.add_argument("--mode",type=str,default="select",help="supported arguments: select, respond.")
     parser.add_argument("--lang",type=str,default="en",help="supported arguments: en, fr.")
     parser.add_argument("--turn_id",type=int,default=1,help="current number of chat turns. By default the conversation is regarded as single-turn.")
     parser.add_argument("--prompt_type",default="standard",type=str,help="supported arguments: standard, AT-standard, CoT, AT-CoT.")
     parser.add_argument("--save_as_csv",action="store_true")
     parser.add_argument("--view_prompt",action="store_true",help="if set, will not load LLM but only print out prompts (saved under --prompt_dir).")
     parser.add_argument("--dry_run",type=str2bool,help="if run a quick run to test.")
     parser.add_argument("--dry_run_number_of_examples",type=int,default=5,help="if set to N, codes will be executed on N first examples in the dataset for a quick test.")
     parser.add_argument("--maximum_retry_times",type=int,default=10,help="in case where LLM fails to output in the speicied format, retry with the same LLM parameters for a limited "
                                                                         "number of times; the last time set do_sample=False.")
     parser.add_argument("--seed",type=int,default=55)

     group = parser.add_argument_group('--llm_options')
     group.add_argument("--batch_size",type=int,default=2,help="batch size used for inference")
     group.add_argument("--max_new_tokens",type=int,default=1000)
     group.add_argument("--temperature",type=float,default=0.6)
     group.add_argument("--no_sampling",action="store_true")
     group.add_argument("--sampling_strategy",type=str,default="top_k",help="sampling strategy for decoding: top_k or top_p. Top_k by default.")
     group.add_argument("--top_k",type=int,default=10)
     group.add_argument("--top_p",type=float,default=0.9)
     group = parser.add_argument_group('--gpu_options')
     group.add_argument("--gpu_partition",type=str)
     group.add_argument("--gpu_node",type=str)

     group = parser.add_argument_group("--ir")
     group.add_argument("--prebuilt_index_name",type=str,default="msmarco-passage")
     group.add_argument("--stage",type=str,default="retrieve+rerank")
     group.add_argument("--k",type=int,default="number of documents returned by BM25.")

     args = parser.parse_args()
     return args