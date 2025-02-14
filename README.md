### Clarifying Ambiguities: On The Role of Ambiguity Types in Prompting Methods for Clarification Generation

### Part 1. Clarification 
💡 Run the following command to prompt LLMs for clarification generation
```
python3 main.py --dataset_name qulac --prompt_type standard --lang en --mode select --dry_run false
```
will generate 5 RQs for each query in Qulac.

💡 You can first verify if prompts are correctly loaded:
```
python3 main.py --dataset_name qulac --prompt_type standard --mode select --view_prompt
```

💡 To run on a gpu cluster, set your GPU partition and nodelist in exp.sh, then run:
```
sh exp.sh qulac en standard false
```

💡 Change the following arguments to fulfill your need:

| Argument      | Description                          | Accepted Values |
|--------------|--------------------------------------|----------------|
| `--prompt_type`     | Prompting method  | `standard`, `AT-standard`, `CoT`, `AT-CoT` |
| `--mode`   | Interaction mode                       | `select`, `respond` |
| `--lang`    | Language                       | `en`, `fr` |
| inference hyperparameters | top_k, temperature, etc. Check python3 main.py --help for complete list of arguments. | |

💡 Use your own data by creating a .json file as example-en.json or example-fr.json under /data/.

### Part 2. Information Retrieval 
💡 Run the following command to retrieve relevant documents for given queries
```
python3 retrieve.py -dataset_name msmarco_dev_small --k 100 --stage retrieve+rerank
```

💡Available hyperparameters 
| Argument      | Description                          | Accepted Values |
|--------------|--------------------------------------|----------------|
| `--k`     | Number of documents to return by BM25.   | `standard`, `AT-standard`, `CoT`, `AT-CoT` |
| `--stage`   | Retrieval pipeline: retrieve only or retrieve+rerank  | `retrieve`, `retrieve+rerank` |