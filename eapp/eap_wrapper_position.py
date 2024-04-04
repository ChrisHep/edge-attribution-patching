import gc
from functools import partial
from typing import Callable, List, Union

import einops
import torch
import tqdm
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from eapp.eap_graph_position import EAPGraph
import sys

def EAP_clean_forward_hook(
    activations: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    upstream_activations_difference: Float[Tensor, "batch_size seq_len n_upstream_nodes d_model"], 
    graph: EAPGraph
):
    upstream_activations_difference = upstream_activations_difference.to(activations.device)
    hook_slice = graph.get_hook_slice(hook.name)
    if activations.ndim == 3:
        # We are in the case of a residual layer or MLP
        # Activations have shape [batch_size, seq_len, d_model]
        # We need to add an extra dimension to make it [batch_size, seq_len, 1, d_model]
        # The hook slice is a slice of length 1
        upstream_activations_difference[:, :, hook_slice, :] = -activations.unsqueeze(-2)
    elif activations.ndim == 4:
        # We are in the case of an attention layer
        # Activations have shape [batch_size, seq_len, n_heads, d_model]
        upstream_activations_difference[:, :, hook_slice, :] = -activations # clean - corrupted

def EAP_corrupted_forward_hook(
    activations: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    upstream_activations_difference: Float[Tensor, "batch_size seq_len n_upstream_nodes d_model"], 
    graph: EAPGraph
):
    upstream_activations_difference = upstream_activations_difference.to(activations.device)
    hook_slice = graph.get_hook_slice(hook.name)
    if activations.ndim == 3:
        upstream_activations_difference[:, :, hook_slice, :] += activations.unsqueeze(-2)
    elif activations.ndim == 4:
        upstream_activations_difference = upstream_activations_difference.to(activations.device)
        upstream_activations_difference[:, :, hook_slice, :] += activations

def EAP_corrupted_backward_hook(
    grad: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    upstream_activations_difference: Float[Tensor, "batch_size seq_len n_upstream_nodes d_model"],
    graph: EAPGraph, 
    result_position: int
):
    upstream_activations_difference = upstream_activations_difference.to(grad.device)
    #upstream_activations_difference = upstream_activations_difference[:, result_position, :, :]
    hook_slice = graph.get_hook_slice(hook.name)

    # we get the slice of all upstream nodes that come before this downstream node
    earlier_upstream_nodes_slice = graph.get_slice_previous_upstream_nodes(hook)

    # grad has shape [batch_size, seq_len, n_heads, d_model] or [batch_size, seq_len, d_model]
    # we want to multiply it by the upstream activations difference
    if grad.ndim == 3:
        grad_expanded = grad.unsqueeze(-2)  # Shape: [batch_size, seq_len, 1, d_model]
        #grad_expanded = grad_expanded[:, result_position, :, :]
    else:
        grad_expanded = grad  # Shape: [batch_size, seq_len, n_heads, d_model]
        # get the slice corresponding to the ICD result position
        #grad_expanded = grad[:, result_position, :, :]

    result_per_position1 = torch.matmul(
        upstream_activations_difference[:, :, earlier_upstream_nodes_slice],
        grad_expanded.transpose(-1, -2)
    ).sum(dim=0) # we sum over the batch_size dimension
    #print(f"result_per_position1: {result_per_position1.shape}")

    # get the slice corespoinding to the ICD result position
    result_per_position = result_per_position1[result_position, :, :]
    # delete the first dimension
    result_per_position = result_per_position.squeeze(0)

    # result_per_position = torch.matmul(
    #     upstream_activations_difference[:, result_position, earlier_upstream_nodes_slice],
    #     grad_expanded.transpose(-1, -2)[:, result_position, :]
    # ).sum(dim=0) # we sum over the batch_size dimension
    #print(f"result_per_position: {result_per_position.shape}")

    graph.eap_scores_per_position = graph.eap_scores_per_position.to(result_per_position.device)

    graph.eap_scores_per_position[earlier_upstream_nodes_slice, hook_slice] += result_per_position


def EAP(
    model: HookedTransformer,
    clean_tokens: Int[Tensor, "batch_size seq_len"],
    corrupted_tokens: Int[Tensor, "batch_size seq_len"],
    metric: Callable,
    upstream_nodes: List[str]=None,
    downstream_nodes: List[str]=None,
    batch_size: int=1,
    alt_attention_mask: Int[Tensor, "batch_size seq_len"]=None,
    base_attention_mask: Int[Tensor, "batch_size seq_len"]=None,
    result_position: List[int]=None
):

    graph = EAPGraph(model.cfg, upstream_nodes, downstream_nodes)

    assert clean_tokens.shape == corrupted_tokens.shape, "Shape mismatch between clean and corrupted tokens"
    num_prompts, seq_len = clean_tokens.shape[0], clean_tokens.shape[1]

    assert num_prompts % batch_size == 0, "Number of prompts must be divisible by batch size"

    upstream_activations_difference = torch.zeros(
        (batch_size, seq_len, graph.n_upstream_nodes, model.cfg.d_model),
        device=model.cfg.device,
        dtype=model.cfg.dtype,
        requires_grad=False
    )

    # set the EAP scores to zero
    graph.reset_scores()

    upstream_hook_filter = lambda name: name.endswith(tuple(graph.upstream_hooks))
    downstream_hook_filter = lambda name: name.endswith(tuple(graph.downstream_hooks))

    corrupted_upstream_hook_fn = partial(
        EAP_corrupted_forward_hook,
        upstream_activations_difference=upstream_activations_difference,
        graph=graph
    )

    clean_upstream_hook_fn = partial(
        EAP_clean_forward_hook,
        upstream_activations_difference=upstream_activations_difference,
        graph=graph
    )

    # corrupted_downstream_hook_fn = partial(
    #     EAP_corrupted_backward_hook,
    #     upstream_activations_difference=upstream_activations_difference,
    #     graph=graph,
    # )
    def create_hook_wrapper(result_positions, upstream_activations_difference, graph, idx, batch_size):
        def hook_wrapper(grad, hook: HookPoint, idx=idx, batch_size=batch_size):
            # Extract the current result_position based on the hook's call context
            # This assumes there's a way to determine the current batch index or similar
            result_position = result_positions[idx:idx+batch_size][0]

            # Call the actual hook function with the dynamically determined result_position
            EAP_corrupted_backward_hook(
                grad,
                hook,
                upstream_activations_difference,
                graph,
                result_position=result_position
            )

        return hook_wrapper

    for idx in tqdm(range(0, num_prompts, batch_size)):
        # we first perform a forward pass on the corrupted input 
        model.add_hook(upstream_hook_filter, clean_upstream_hook_fn, "fwd")
        
        # we don't need gradients for this forward pass
        # we'll take the gradients when we perform the forward pass on the clean input
        with torch.no_grad(): 
            clean_tokens = clean_tokens.to(model.cfg.device)
            base_attention_mask = base_attention_mask.to(model.cfg.device)
            model(clean_tokens[idx:idx+batch_size], return_type=None, attention_mask=base_attention_mask[idx:idx+batch_size])

        # now we perform a forward and backward pass on the corrupted input
        model.reset_hooks()
        corrupted_downstream_hook_fn = create_hook_wrapper(result_position, upstream_activations_difference, graph, idx, batch_size)
        model.add_hook(upstream_hook_filter, corrupted_upstream_hook_fn, "fwd")
        model.add_hook(downstream_hook_filter, corrupted_downstream_hook_fn, "bwd")

        corrupted_tokens = corrupted_tokens.to(model.cfg.device)
        alt_attention_mask = alt_attention_mask.to(model.cfg.device)
        value = metric(model(corrupted_tokens[idx:idx+batch_size], return_type="logits", attention_mask=alt_attention_mask[idx:idx+batch_size]))
        value.backward()
        
        # We delete the activation differences tensor to free up memory
        model.zero_grad()
        upstream_activations_difference *= 0

    del upstream_activations_difference
    gc.collect()
    torch.cuda.empty_cache()
    model.reset_hooks()

    graph.eap_scores_per_position /= num_prompts
    graph.eap_scores_per_position = graph.eap_scores_per_position.cpu()

    return graph