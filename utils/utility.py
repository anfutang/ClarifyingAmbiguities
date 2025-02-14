import os
import datetime
import numpy as np
import random
import torch
import transformers

valid_prompt_types = ["standard","AT-standard","CoT","AT-CoT"]

def validate_arguments(args):
    assert os.path.exists(args.data_dir), "input data does not exist."
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir,exist_ok=True)
    if not os.path.exists(args.logging_dir):
        os.makedirs(args.logging_dir,exist_ok=True)
    if not os.path.exists(args.prompt_dir):
        os.makedirs(args.prompt_dir,exist_ok=True)
    assert args.turn_id >= 1, "turn id must be superior or equal to 1."
    assert args.mode in ["select","respond",], "INVALID mode."
    assert args.prompt_type in valid_prompt_types, "INVALID prompt type."

def build_dst_folder(args,ir=False):
    if not ir:
        comb = ["clarification", args.dataset_name, args.mode, args.prompt_type]

        if args.view_prompt:
            curr_logging_dir, curr_output_dir = args.prompt_dir, args.prompt_dir
        else:
            curr_logging_dir, curr_output_dir = args.logging_dir, args.output_dir

        for c in comb:
            curr_logging_dir = os.path.join(curr_logging_dir,c)
            if not os.path.exists(curr_logging_dir):
                os.makedirs(curr_logging_dir,exist_ok=True)
            curr_output_dir = os.path.join(curr_output_dir,c)
            if not os.path.exists(curr_output_dir):
                os.makedirs(curr_output_dir,exist_ok=True)
    else:
        comb = ["ir", args.dataset_name]
        curr_logging_dir, curr_output_dir = args.logging_dir, args.output_dir

        for c in comb:
            curr_logging_dir = os.path.join(curr_logging_dir,c)
            if not os.path.exists(curr_logging_dir):
                os.makedirs(curr_logging_dir,exist_ok=True)
            curr_output_dir = os.path.join(curr_output_dir,c)
            if not os.path.exists(curr_output_dir):
                os.makedirs(curr_output_dir,exist_ok=True)
    
    return curr_logging_dir, curr_output_dir

def show_job_infos(args,ir=False):
    infos = []
    if not ir:
        infos.append('\n'+'='*5 + "Job information" + '='*5)
        infos.append("*Start time:"+str(datetime.datetime.now()))
        infos.append(f"*Gpu: {args.gpu_partition}-{args.gpu_node}")
        infos.append(f"*Dataset: {args.dataset_name}")
        infos.append(f"*Turn: {args.turn_id}")
        infos.append(f"*Maximum retry times: {args.maximum_retry_times}")
        infos.append(f"*Lanuage: {args.lang}")
        infos.append(f"*Mode: {args.mode}")
        infos.append(f"*Prompt type: {args.prompt_type}")
        infos.append('='*10+'\n')
    else:
        infos.append('\n'+'='*5 + "Job information" + '='*5)
        infos.append("*Start time:"+str(datetime.datetime.now()))
        infos.append(f"*Gpu: {args.gpu_partition}-{args.gpu_node}")
        infos.append(f"*Information retrieval: yes")
        infos.append(f"*Dataset: {args.dataset_name}")
        infos.append(f"*Prebuilt index: {args.prebuilt_index_name}")
        infos.append('='*10+'\n')
    return '\n'.join(infos)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)
