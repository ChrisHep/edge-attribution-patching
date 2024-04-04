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
    
        self.intervention_list = intervention_list
        self.tokenizer = tokenizer
        self.device = device
        self.base_string_toks = []
        self.alt_string_toks = []
        self.res_base_toks = []
        self.pred_res_alt_toks = []
        self.result_position = []

    def __len__(self):
        return len(self.intervention_list)
    
    def __getitem__(self, idx):
        return (self.base_string_toks[idx], self.alt_string_toks[idx], 
                self.res_base_toks[idx], self.pred_res_alt_toks[idx],
                self.result_position[idx])
    
    def __iter__(self):
        return iter(self.intervention_list)

    

    def group_tensors_by_length(self, base_string_toks):
        # Dictionary to hold grouped tensors by their length
        grouped_tensors_by_length = {}
        # Keep track of the original indices for reordering
        reorder_indices = []

        for i, tensor in enumerate(base_string_toks):
            length = tensor.size(1)  # Get the length of the tensor (second dimension)
            if length not in grouped_tensors_by_length:
                grouped_tensors_by_length[length] = []
            grouped_tensors_by_length[length].append(tensor)
            reorder_indices.append((length, len(grouped_tensors_by_length[length]) - 1, i))

        # Sort reorder_indices by length and then by their order of appearance
        reorder_indices.sort(key=lambda x: (x[0], x[1]))
        # Extract the final order of indices to match the grouped order
        final_order_indices = [index for _, _, index in reorder_indices]

        # Convert the dictionary to a list of groups
        grouped_tensors = [group for _, group in sorted(grouped_tensors_by_length.items())]

        return grouped_tensors, final_order_indices

    
    def reorder_list_according_to_indices(self, original_list, indices):
        return [original_list[i] for i in indices]
    
    # padding based solution
    def create_intervention_dataset(self):
        for intervention in self.intervention_list:
            self.base_string_toks.append(intervention.base_string_tok.T.flip([0]))
            self.alt_string_toks.append(intervention.alt_string_tok.T)
            self.res_base_toks.append(intervention.res_base_tok[0])
            self.pred_res_alt_toks.append(intervention.pred_res_alt_tok[0])
            if len(intervention.base_string_tok_list) == 14:
                self.result_position.append(16)
            elif len(intervention.base_string_tok_list) == 20:
                self.result_position.append(14)
            else:
                self.result_position.append(11)
        self.base_string_toks = t.nn.utils.rnn.pad_sequence([i for i in self.base_string_toks], batch_first = True, padding_value=0).flip(dims=[1])
        self.base_string_toks = t.squeeze(self.base_string_toks, dim=-1)
        self.alt_string_toks = t.nn.utils.rnn.pad_sequence([i for i in self.alt_string_toks], batch_first = True, padding_value=0).flip(dims=[1])
        self.alt_string_toks = t.squeeze(self.alt_string_toks, dim=-1)

        self.base_attention_mask = (self.base_string_toks != 0).long()
        self.alt_attention_mask = (self.alt_string_toks != 0).long()

    def shuffle(self):
        t.manual_seed(0)
        indices = t.randperm(len(self.base_string_toks))
        self.base_string_toks = self.base_string_toks[indices]
        self.alt_string_toks = self.alt_string_toks[indices]
        self.res_base_toks = [self.res_base_toks[i] for i in indices]
        self.pred_res_alt_toks = [self.pred_res_alt_toks[i] for i in indices]
        self.base_attention_mask = self.base_attention_mask[indices]
        self.alt_attention_mask = self.alt_attention_mask[indices]
        self.result_position = [self.result_position[i] for i in indices]
        