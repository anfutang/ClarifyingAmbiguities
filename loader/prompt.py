import os
import sys
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from parser.parser_templates import *
from .prompt_templates import *
from .system_instructions import SystemInstruction

logger = logging.getLogger(__name__)

split_str = '\n'*3 + '='*10 + '\n'*3

class PromptLoader:
    def __init__(self,args,tokenizer):
        self.dataset_name = args.dataset_name
        self.prompt_type = args.prompt_type
        self.prompt_dir = args.prompt_dir
        self.mode = args.mode
        self.turn_id = args.turn_id
        self.lang = args.lang
        self.view_prompt = args.view_prompt
        self.tokenizer = tokenizer
        self.load_few_shot_examples()
        self.get_prompt_template() 
        self.get_parser()
        self.get_system_instruction(args)
        self.get_few_shot_examples()
        self.final_prompt_template = PromptTemplate(template=self.prompt_template,
                                                    input_variables=self.input_variables,
                                                    partial_variables={"system_instruction":self.system_instruction,
                                                                       "format_instruction": self.parser.get_format_instructions(),
                                                                       "few_shot_examples":self.few_shot_examples})
    
    def get_prompt_template(self):
        if self.lang == "en":
            self.prompt_template = generation_template_en
        else:
            self.prompt_template = generation_template_fr
        self.input_variables = ["query"]
        self.prompt_template = self.tokenizer.apply_chat_template(self.prompt_template,add_generation_prompt=True,tokenize=False)

    def get_parser(self):
        if self.mode == "select":
            if "CoT" in self.prompt_type:
                pydantic_obj = RQCoTMultiple
            else:
                pydantic_obj = RQMultiple
        else:
            if "CoT" in self.prompt_type:
                pydantic_obj = CQCoTSingle
            else:
                pydantic_obj = CQSingle
        
        self.parser = PydanticOutputParser(pydantic_object=pydantic_obj)

    def get_system_instruction(self,args):
        self.system_instruction = SystemInstruction(args).instruction

    def get_few_shot_examples(self):
        self.few_shot_examples = self.all_few_shot_examples[self.lang][self.mode][self.prompt_type]
        
    def load_few_shot_examples(self):
        all_few_shot_example_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',"few_shot_examples"))
        self.all_few_shot_examples = json.load(open(os.path.join(all_few_shot_example_dir,"examples.json")))

    def format(self,data):
        prompts = []
        for q in data["query"]:
            prompts.append(self.final_prompt_template.format_prompt(query=q).text)

        if self.view_prompt:
            prompt_dir = os.path.join(self.prompt_dir,self.dataset_name,self.mode,self.prompt_type)
            with open(os.path.join(prompt_dir,f"prompts.txt"),'w') as f:
                f.write(split_str.join(prompts))
            logger.info(f"VIEW PROMPT mode: prompt texts are printed out and saved under {prompt_dir}.")
        return prompts