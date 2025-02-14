### Clarifying Ambiguities: On The Role of Ambiguity Types in Prompting Methods for Clarification Generation

Run the following command
```
python3 main.py --dataset_name qulac --prompt_type standard --lang en --mode select --dry_run false
```
will generate 5 RQs for each query in Qulac.

You can first verify if prompts are correctly loaded:
```
python3 main.py --dataset_name qulac --prompt_type standard --mode select --view_prompt
```

To run on a gpu cluster, set your GPU partition and nodelist in exp.sh, then run:
```
sh exp.sh qulac en standard false
```

Change the following arguments to fulfill your need:
- prompt_type: standard, AT-standard, CoT, AT-CoT
- mode: select, respond
- lang: en, fr
(inference hypermeters: top_k, temperature, etc. Check python3 main.py --help for all available options.)

Use your own data by creating a .json file as example-en.json or example-fr.json under /data/.

