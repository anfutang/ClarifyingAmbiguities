import os
import re
import json

def clean_sentence(s):
    return re.sub(r'^\(\d+\)\s*', '', s)
    
class DataLoader:
    def __init__(self,args):
        self.dataset_name = args.dataset_name
        self.turn_id = args.turn_id
        self.mode = args.mode
        self.prompt_type = args.prompt_type
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','data','clarification'))
        # self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','output'))
        self.load()

    def load(self):
        self.data = {}
        source_data = json.load(open(os.path.join(self.data_dir,f"{self.dataset_name}.json")))
        self.data["query"] = source_data["query"]

    def extend_data_based_on_user_intention(self,uis,data):
        extended_data = []
        for d, ui in zip(data,uis):
            extended_data += [d] * len(ui)
        return extended_data
    
    def flatten_user_intention(self,uis):
        flattened_uis = []
        for ui in uis:
            flattened_uis += ui
        return flattened_uis

