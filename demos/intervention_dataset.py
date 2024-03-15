import interventions
import pandas as pd
from typing import Union, List, Optional
import warnings
import torch as t
import numpy as np
from transformers import AutoTokenizer
import random
import copy
import re
from rich import print as rprint
from rich.table import Table

class InterventionDataset:
    def __init__(self, 
                 intervention_list,
                 device,
                 tokenizer: AutoTokenizer):
        """
        Args:
            data (Union[pd.DataFrame, str]): The data to be used for the dataset. If a string is passed, it is assumed to be a path to a csv file.
            tokenizer (AutoTokenizer): The tokenizer to be used for the dataset.
            max_length (int, optional): The maximum length of the input sequence. Defaults to 512.
            label_column (str, optional): The column in the dataframe that contains the labels. Defaults to 'label'.
            text_column (str, optional): The column in the dataframe that contains the text. Defaults to 'text'.
            intervention_column (Optional[str], optional): The column in the dataframe that contains the interventions. If None, interventions will be generated. Defaults to None.
            interventions (Optional[List[interventions.Intervention]], optional): A list of interventions to be used. If None, interventions will be generated. Defaults to None.
            intervention_prob (float, optional): The probability of applying an intervention. Defaults to 0.5.
            intervention_max (int, optional): The maximum number of interventions to apply. Defaults to 3.
            intervention_min (int, optional): The minimum number of interventions to apply. Defaults to 1.
            intervention_seed (int, optional): The seed to be used for generating interventions. Defaults to 42.
        """
        self.intervention_list = intervention_list
        self.tokenizer = tokenizer
        self.device = device
        self.base_string_toks = []
        self.alt_string_toks = []
        self.res_base_toks = []
        self.pred_res_alt_toks = []

    def __len__(self):
        return len(self.intervention_list)
    
    def __getitem__(self, idx):
        return (self.base_string_toks[idx], self.alt_string_toks[idx], self.res_base_toks[idx], self.pred_res_alt_toks[idx])
    
    def __iter__(self):
        return iter(self.intervention_list)
    
    def create_intervention_dataset(self):
        for intervention in self.intervention_list:
            self.base_string_toks.append(intervention.base_string_tok.to(self.device))
            self.alt_string_toks.append(intervention.alt_string_tok.to(self.device))
            self.res_base_toks.append(intervention.res_base_tok[0])
            self.pred_res_alt_toks.append(intervention.pred_res_alt_tok[0])

        self.base_string_toks = t.vstack(self.base_string_toks)
        self.alt_string_toks = t.vstack(self.alt_string_toks)