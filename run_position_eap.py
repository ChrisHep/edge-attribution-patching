import torch as t
from torch import Tensor
import einops
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import numpy as np
from eapp.eap_wrapper_position import EAP
from jaxtyping import Float
import yaml
import pickle
import os
from demos import intervention_dataset
import hydra
import wandb
import json
from omegaconf import DictConfig, OmegaConf
import sys
device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

def ave_logit_difference(
    logits: Float[Tensor, 'batch seq d_vocab'],
    intervention_dataset,
    per_prompt: bool = False
):
    batch_size = logits.size(0)
    correct_logits = logits[range(batch_size), -1, intervention_dataset.res_base_toks[:batch_size]]
    incorrect_logits = logits[range(batch_size), -1, intervention_dataset.pred_res_alt_toks[:batch_size]]
    logit_diff = correct_logits - incorrect_logits
    return logit_diff if per_prompt else logit_diff.mean()


def logits_in_batches(model, tokens, attn_mask, bsize):
    model.eval()
    seq_len = tokens.size(0)
    all_logits = []

    with t.no_grad():
        for i in range(0, seq_len, bsize):
            input = tokens[i:i+bsize].to(model.cfg.device)
            attn_mask = attn_mask[i:i+bsize].to(model.cfg.device)
            logits = model(input=input, attention_mask=attn_mask)
            logits = logits.detach().cpu()
            input = input.detach().cpu()
            attn_mask = attn_mask.detach().cpu()
            all_logits.append(logits)
    return t.cat(all_logits, dim=0)


@hydra.main(config_path="conf", config_name="config")
def run_exp(args: DictConfig):
    print(OmegaConf.to_yaml(args))


    file_name_arabic = args.data_dir
    file_name_arabic += '/' + str(args.model)
    file_name_arabic += '/intervention_' + str(args.n_shots) + '_shots_max_' + str(args.max_n) + '_' + 'arabic'
    file_name_arabic += '_further_templates' if args.extended_templates else ''
    file_name_arabic += '_acdc' if args.acdc_data else ''
    file_name_arabic += '.pkl'

    with open(file_name_arabic, 'rb') as f:
        intervention_list_arabic = pickle.load(f)
    print("Loaded data from", file_name_arabic)
    if args.debug_run:
        intervention_list_arabic = intervention_list_arabic[-2:]

    file_name_words = args.data_dir
    file_name_words += '/' + str(args.model)
    file_name_words += '/intervention_' + str(args.n_shots) + '_shots_max_' + str(args.max_n) + '_' + 'words'
    file_name_words += '_further_templates' if args.extended_templates else ''
    file_name_words += '_acdc' if args.acdc_data else ''
    file_name_words += '.pkl'

    with open(file_name_words, 'rb') as f:
        intervention_list_words = pickle.load(f)
    print("Loaded data from", file_name_words)
    if args.debug_run:
        intervention_list_words = intervention_list_words[-2:]

    intervention_list = intervention_list_arabic + intervention_list_words

    k_edges = int(args.k_edges)

    wandb_name = f"{args.model}_{args.n_shots}_shots_max_{args.max_n}"

    wandb.init(project="eap", name=wandb_name, notes='', mode=args.wandb_mode)
    args_to_log = dict(args)
    print("\n" + json.dumps(str(args_to_log), indent=4) + "\n")
    wandb.config.update(args_to_log)


    # load model
    model = HookedTransformer.from_pretrained(
    args.model,
    center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device=device,
    n_devices=7,
    move_to_device=True,
    dtype='float16'
    )

    print(f"Loaded model {model}")

    model.set_use_hook_mlp_in(True)
    model.set_use_split_qkv_input(True)
    model.set_use_attn_result(True)
    model.tokenizer.padding_side = "left"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"Using tokenizer {tokenizer}")

    intervention_data = intervention_dataset.InterventionDataset(intervention_list, device, model.tokenizer)
    intervention_data.create_intervention_dataset()
    intervention_data.shuffle()

    clean_logits = logits_in_batches(model, intervention_data.base_string_toks, intervention_data.base_attention_mask, 9)
    corrupt_logits = logits_in_batches(model, intervention_data.alt_string_toks, intervention_data.alt_attention_mask, 9)
    clean_logit_diff = ave_logit_difference(clean_logits, intervention_data, per_prompt=False).item()
    corrupt_logit_diff = ave_logit_difference(corrupt_logits, intervention_data, per_prompt=False).item()
    
    if args.normalized_metric:
        def metric(
            logits: Float[Tensor, "batch seq_len d_vocab"],
            corrupted_logit_diff: float = corrupt_logit_diff,
            clean_logit_diff: float = clean_logit_diff,
            intervention_dataset: intervention_data = intervention_data,
            per_prompt: bool = False
        ):
            patched_logit_diff = ave_logit_difference(logits, intervention_dataset, per_prompt)
            metric_result = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
            return metric_result
    
        clean_metric = metric(clean_logits, corrupt_logit_diff, clean_logit_diff, intervention_data, per_prompt=False)
        corrupt_metric = metric(corrupt_logits, corrupt_logit_diff, clean_logit_diff, intervention_data, per_prompt=False)

    else:
        def metric(
            logits: Float[Tensor, "batch seq_len d_vocab"],
            corrupted_logit_diff: float = corrupt_logit_diff,
            intervention_dataset: intervention_data = intervention_data,
            per_prompt: bool = False
        ):
            patched_logit_diff = ave_logit_difference(logits, intervention_dataset, per_prompt)
            metric_result = patched_logit_diff - corrupted_logit_diff
            return metric_result
        
        clean_metric = metric(clean_logits, corrupt_logit_diff, intervention_data, per_prompt=False)
        corrupt_metric = metric(corrupt_logits, corrupt_logit_diff, intervention_data, per_prompt=False)
        
    print(f"clean metric: {clean_metric}, corrupt metric: {corrupt_metric}")

    
    model.reset_hooks()

    graph = EAP(
        model,
        clean_tokens=intervention_data.base_string_toks,
        corrupted_tokens=intervention_data.alt_string_toks,
        metric=metric,
        upstream_nodes=["mlp", "head"],
        downstream_nodes=["mlp", "head"],
        batch_size=1,
        alt_attention_mask=intervention_data.alt_attention_mask,
        base_attention_mask=intervention_data.base_attention_mask,
        result_position=intervention_data.result_position
    )   

    top_edges = graph.top_edges(n=k_edges, abs_scores=False)
    for from_edge, to_edge, score in top_edges:
        print(f'{from_edge} -> [{round(score, 3)}] -> {to_edge}')

    graph.show(n=k_edges, normalized_metric=args.normalized_metric)
    



if __name__ == '__main__':
    run_exp()

