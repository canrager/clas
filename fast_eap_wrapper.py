import torch
import gc
import numpy as np
from transformer_lens import HookedTransformer, ActivationCache
from torch import Tensor
from jaxtyping import Float, Bool, Int
from typing import List, Callable, Tuple, Union, Dict, Optional
import einops

class ModelWrapper:
    def __init__(self, model):
        self.model = model

        self.model.set_use_hook_mlp_in(True)
        self.model.set_use_split_qkv_input(True)
        self.model.set_use_attn_result(True)

        self.n_layers = self.model.cfg.n_layers
        self.n_heads = self.model.cfg.n_heads
        self.d_model = self.model.cfg.d_model
        self.dtype = self.model.cfg.dtype
        self.device = self.model.cfg.device

        self.valid_upstream_node_types = ["resid_pre", "mlp", "head"]
        self.valid_downstream_node_types = ["resid_post", "mlp", "head"]

        self.valid_upstream_hook_types = ["hook_resid_pre", "hook_result", "hook_mlp_out"]
        self.valid_downstream_hook_types = ["hook_q_input", "hook_k_input", "hook_v_input", "hook_mlp_in", "hook_resid_post"]

        self.upstream_nodes = []
        self.downstream_nodes = []

        self.upstream_node_index: Dict[str, int] = {}
        self.downstream_node_index: Dict[str, int] = {}

        self.upstream_hook_slice: Dict[str, slice] = {}
        self.downstream_hook_slice: Dict[str, slice] = {}

        self.upstream_nodes_before_layer: Dict[int, slice] = {}
        self.upstream_nodes_before_attn_layer: Dict[int, slice] = {}
        self.upstream_nodes_before_mlp_layer: Dict[int, slice] = {}

    def reset_graph(self):
        self.upstream_nodes = []
        self.downstream_nodes = []

        self.upstream_node_index: Dict[str, int] = {}
        self.downstream_node_index: Dict[str, int] = {}

        self.upstream_hook_slice: Dict[str, slice] = {} 
        self.downstream_hook_slice: Dict[str, slice] = {}

        self.upstream_nodes_before_layer: Dict[int, slice] = {}
        self.upstream_nodes_before_mlp_layer: Dict[int, slice] = {}

    def setup_edges(self, upstream_nodes=None, downstream_nodes=None):
        # we first reset all the graph-related attributes
        self.reset_graph()

        # if no nodes are specified, we assume that all of them will be used
        if upstream_nodes is None:
            upstream_nodes = self.valid_upstream_node_types.copy()
        
        if downstream_nodes is None:
            downstream_nodes = self.valid_downstream_node_types.copy()

        # we can assume that the two lists of hooks are sorted by layer number
        self.upstream_hooks, self.downstream_hooks = self.get_hooks_from_nodes(upstream_nodes, downstream_nodes)

        upstream_node_index = 0

        for hook_name in self.upstream_hooks:
            layer = int(hook_name.split(".")[1])
            hook_type = hook_name.split(".")[-1]
            
            # we store the slice of all upstream nodes previous to this layer
            if layer not in self.upstream_nodes_before_layer:
                # we must check previous layers too because we might have skipped some
                for earlier_layer in range(0, layer + 1):
                    if earlier_layer not in self.upstream_nodes_before_layer:
                        self.upstream_nodes_before_layer[earlier_layer] = slice(0, upstream_node_index)
                        self.upstream_nodes_before_attn_layer[layer] = slice(0, upstream_node_index)
                        self.upstream_nodes_before_mlp_layer[layer] = slice(0, upstream_node_index)

            if hook_type == "hook_resid_pre":
                self.upstream_nodes.append(f"resid_pre.{layer}")
                self.upstream_node_index[f"resid_pre.{layer}"] = upstream_node_index
                self.upstream_hook_slice[hook_name] = slice(upstream_node_index, upstream_node_index + 1)
                self.upstream_nodes_before_attn_in_layer[layer] = slice(0, upstream_node_idx + 1)
                upstream_node_index += 1

            elif hook_type == "hook_result":
                for head_idx in range(self.model.cfg.n_heads):
                    self.upstream_nodes.append(f"head.{layer}.{head_idx}")
                    self.upstream_node_index[f"head.{layer}.{head_idx}"] = upstream_node_index + head_idx
                self.upstream_hook_slice[hook_name] = slice(upstream_node_index, upstream_node_index + self.n_heads)
                self.upstream_nodes_before_mlp_in_layer[layer] = slice(0, upstream_node_idx + self.n_heads)
                upstream_node_index += self.n_heads 

            elif hook_type == "hook_mlp_out":
                self.upstream_nodes.append(f"mlp.{layer}")
                self.upstream_node_index[f"mlp.{layer}"] = upstream_node_index
                self.upstream_hook_slice[hook_name] = slice(upstream_node_index, upstream_node_index + 1)
                upstream_node_index += 1

            else:
                raise NotImplementedError("Invalid upstream hook type")

        # if there are no more upstream nodes after a certain layer we still have
        # to save that into the slice dictionaries
        for layer in range(0, self.n_layers):
            if layer not in self.upstream_nodes_before_layer:
                self.upstream_nodes_before_layer[layer] = slice(0, upstream_node_index)
                self.upstream_nodes_before_attn_layer[layer] = slice(0, upstream_node_index)
                self.upstream_nodes_before_mlp_layer[layer] = slice(0, upstream_node_index)

        downstream_node_index = 0

        for hook_name in self.downstream_hooks:
            layer = int(hook_name.split(".")[1])
            hook_type = hook_name.split(".")[-1]

            if hook_type == "hook_q_input" or hook_type == "hook_k_input" or hook_type == "hook_v_input":
                letter = hook_type.split("_")[1].lower()
                for head_idx in range(self.model.cfg.n_heads):
                    self.downstream_nodes.append(f"head.{layer}.{head_idx}.{letter}")
                    self.downstream_node_index[f"head.{layer}.{head_idx}.{letter}"] = downstream_node_index + head_idx
                self.downstream_hook_slice[hook_name] = slice(downstream_node_index, downstream_node_index + self.n_heads)
                downstream_node_index += self.n_heads 

            elif hook_type == "hook_mlp_in":
                self.downstream_nodes.append(f"mlp.{layer}")
                self.downstream_node_index[f"mlp.{layer}"] = downstream_node_index
                self.downstream_hook_slice[hook_name] = slice(downstream_node_index, downstream_node_index + 1)
                downstream_node_index += 1

            elif hook_type == "hook_resid_post":
                self.downstream_nodes.append(f"resid_post.{layer}")
                self.downstream_node_index[f"resid_post.{layer}"] = downstream_node_index
                self.downstream_hook_slice[hook_name] = slice(downstream_node_index, downstream_node_index + 1)
                downstream_node_index += 1

            else:
                raise NotImplementedError("Invalid upstream hook type")

        self.n_upstream_nodes = len(self.upstream_nodes)
        self.n_downstream_nodes = len(self.downstream_nodes)

        activations_tensor_in_gb = self.n_upstream_nodes * self.d_model * self.dtype.itemsize / 2**30 
        print(f"Saving activations requires {activations_tensor_in_gb:.4f} GB of memory per token")

    # given a set of upstream nodes and downstream nodes, this function returns the corresponding hooks
    # to access the activations of these nodes
    # we return the list of hooks sorted by layer number
    def get_hooks_from_nodes(self, upstream_nodes, downstream_nodes):

        # we first check that the types of the nodes passed are valid
        for node in upstream_nodes:
            node_type = node.split(".")[0] # 'resid_pre', 'mlp' or 'head'
            assert node_type in self.valid_upstream_node_types, "Invalid upstream node"

        for node in downstream_nodes:
            node_type = node.split(".")[0] # 'resid_post', 'mlp' or 'head'
            assert node_type in self.valid_downstream_node_types, "Invalid downstream node"

        upstream_hooks = []
        downstream_hooks = []

        for node in upstream_nodes:
            node_is_layer_specific = (len(node.split(".")) > 1)
            if not node_is_layer_specific:
                # we are in the case of a global node that applies to all layers
                hook_type = "hook_resid_pre" if node == "resid_pre" else "hook_mlp_out" if node == "mlp" else "attn.hook_result"
                for layer in range(self.n_layers):
                    upstream_hooks.append(f"blocks.{layer}.{hook_type}")
            else:
                assert node.split(".")[1].isdigit(), "Layer number must be an integer"
                layer = int(node.split(".")[1])

                hook_type = "hook_resid_pre" if node == "resid_pre" else "hook_mlp_out" if node == "mlp" else "attn.hook_result"
                upstream_hooks.append(f"blocks.{layer}.{hook_type}")

        for node in downstream_nodes:
            node_is_layer_specific = (len(node.split(".")) > 1)
            if not node_is_layer_specific:
                # we are in the case of a global node that applies to all layers
                if node == "head":
                    for layer in range(self.n_layers):
                        for letter in "qkv":
                            downstream_hooks.append(f"blocks.{layer}.hook_{letter}_input")
                elif node == "resid_post" or node == "mlp":
                    hook_type = "hook_resid_post" if node == "resid_post" else "hook_mlp_in"
                    for layer in range(self.n_layers):
                        downstream_hooks.append(f"blocks.{layer}.{hook_type}")
                else:
                    raise NotImplementedError("Invalid downstream node")
            else:
                # we are in the case of a node specified for a single layer
                assert node.split(".")[1].isdigit(), "Layer number must be an integer"
                layer = int(node.split(".")[1])

                if node.startswith("resid_post") or node.startswith("mlp"):
                    hook_type = "hook_resid_post" if node.startswith("resid_post") else "hook_mlp_in"
                    downstream_hooks.append(f"blocks.{layer}.{hook_type}")
                elif node.startswith("head"):
                    # in this case a specific head is specified with (optionally) a corresponding input channel (q, k or v)
                    head_idx = int(node.split(".")[2])
                    letters = ["q", "k", "v"]
                    if len(node.split(".")) == 4:
                        # a specific input channel is specified so we modify the hook name accordingly
                        letter_specified = node.split(".")[3]
                        assert letter_specified in letters, "Invalid letter specified"
                        letters = [letter_specified]
                    for letter in letters:
                        downstream_hooks.append(f"blocks.{layer}.hook_{letter}_input")
                else:
                    raise NotImplementedError("Invalid downstream node")

        upstream_hooks = list(set(upstream_hooks))
        downstream_hooks = list(set(downstream_hooks))

        component_ordering = {
            # upstream
            "hook_resid_pre": 0,
            "hook_result": 1,
            "hook_mlp_out": 2,
            # downstream
            "hook_q_input": 0,
            "hook_k_input": 1,
            "hook_v_input": 2,
            "hook_mlp_in": 3,
            "hook_resid_post": 4
        }

        def comparison_fn(hook_a, hook_b):
            layer_a = int(hook_a.split(".")[1])
            layer_b = int(hook_b.split(".")[1])
            if layer_a != layer_b:
                return layer_a < layer_b
            hook_type_a = hook_a.split(".")[-1]
            hook_type_b = hook_b.split(".")[-1]
            assert hook_type_a != hook_type_b, "Hooks are duplicated"
            return component_ordering[hook_type_a] < component_ordering[hook_type_b]

        # we sort the hooks by the order in which they appear in the computation
        upstream_hooks = sorted(upstream_hooks, cmp=hook_comparison_fn)
        downstream_hooks = sorted(downstream_hooks, cmp=hook_comparison_fn)

        return upstream_hooks, downstream_hooks

    def get_slice_previous_upstream_nodes(self, downstream_hook):
        layer = downstream_hook.layer
        hook_type = downstream_hook.name.split(".")[-1]
        if hook_type == "hook_resid_post":
            return self.upstream_nodes_before_layer[layer + 1]
        if hook_type == "hook_mlp_in":
            return self.upstream_nodes_before_mlp_layer[layer]
        elif hook_type in ["hook_q_input", "hook_k_input", "hook_v_input"]:
            return self.upstream_nodes_before_layer[layer]
        else:
            raise NotImplementedError("Invalid upstream hook type")

    def run_eap(self, clean_tokens, corrupted_tokens, metric, upstream_nodes=None, downstream_nodes=None):
        self.setup_edges(upstream_nodes, downstream_nodes)
        
        self.eap_scores = torch.zeros((self.n_upstream_nodes, self.n_downstream_nodes), device=self.device)

        seq_len = clean_tokens.shape[1]
        batch_size = clean_tokens.shape[0]

        assert seq_len == corrupted_tokens.shape[1], "Sequence length mismatch"
        assert batch_size == corrupted_tokens.shape[0], "Batch size mismatch"

        activations_diff_in_gb = batch_size * seq_len * self.n_upstream_nodes * self.d_model * self.dtype.itemsize / 2**30 
        memory_allocated_in_gb = torch.cuda.memory_allocated() / 2**30
        total_memory_in_gb = torch.cuda.get_device_properties(self.device).total_memory / 2**30
        print(f"Saving activation differences requires {activations_diff_in_gb:.2f} GB of memory")
        assert activations_diff_in_gb < total_memory_in_gb - memory_allocated_in_gb, "Not enough memory to store activation differences"

        self.model.reset_hooks()
        gc.collect()
        torch.cuda.empty_cache()

        upstream_activations_difference = torch.zeros((batch_size, seq_len, self.n_upstream_nodes, self.d_model), device=self.device, dtype=self.dtype, requires_grad=False)

        memory_allocated_in_gb = torch.cuda.memory_allocated() / 2**30
        print(f"Total memory allocated after creating activation differences tensor is {memory_allocated_in_gb:.2f} GB out of {total_memory_in_gb:.2f} GB")

        def corrupted_forward_hook(activations, hook):
            hook_slice = self.upstream_hook_slice[hook.name]
            if activations.ndim == 3:
                # we are in the case of a residual layer or MLP
                # activations have shape [batch_size, seq_len, d_model]
                # we need to add a dimension to make it [batch_size, seq_len, 1, d_model]
                # the hook slice is a slice of size 1
                upstream_activations_difference[:, :, hook_slice, :] = -activations.unsqueeze(-2)
            elif activations.ndim == 4:
                # we are in the case of an attention layer
                # activations have shape [batch_size, seq_len, n_heads, d_model]
                upstream_activations_difference[:, :, hook_slice, :] = -activations

        def clean_forward_hook(activations, hook):
            hook_slice = self.upstream_hook_slice[hook.name]
            if activations.ndim == 3:
                upstream_activations_difference[:, :, hook_slice, :] += activations.unsqueeze(-2)
            elif activations.ndim == 4:
                upstream_activations_difference[:, :, hook_slice, :] += activations

        def backward_hook(grad, hook):
            hook_slice = self.downstream_hook_slice[hook.name]

            # we get the slice of all upstream nodes that come before this downstream node
            earlier_upstream_nodes_slice = self.get_slice_previous_upstream_nodes(hook)

            # grad has shape [batch_size, seq_len, n_heads, d_model] or [batch_size, seq_len, d_model]
            # we want to multiply it by the upstream activations difference
            if grad.ndim == 3:
                grad_expanded = grad.unsqueeze(-2)  # Shape: [batch_size, seq_len, 1, d_model]
            else:
                grad_expanded = grad  # Shape: [batch_size, seq_len, n_heads, d_model]

            # we compute the mean over the batch_size and seq_len dimensions
            result = torch.matmul(
                upstream_activations_difference[:, :, earlier_upstream_nodes_slice],
                grad_expanded.transpose(-1, -2)
            ).sum(dim=0).sum(dim=0) # we sum over the batch_size and seq_len dimensions

            self.eap_scores[earlier_upstream_nodes_slice, hook_slice] = result 

        upstream_hook_filter = lambda name: name.endswith(tuple(self.upstream_hooks))
        downstream_hook_filter = lambda name: name.endswith(tuple(self.downstream_hooks))

        # we first perform a forward pass on the corrupted input 
        self.model.add_hook(upstream_hook_filter, corrupted_forward_hook, "fwd")

        # we don't need gradients for this forward pass
        # we'll take the gradients when we perform the forward pass on the clean input
        with torch.no_grad(): 
            corrupted_tokens = corrupted_tokens.to(self.device)
            self.model(corrupted_tokens, return_type=None)        

        # now we perform a forward and backward pass on the clean input
        self.model.reset_hooks()
        self.model.add_hook(upstream_hook_filter, clean_forward_hook, "fwd")
        self.model.add_hook(downstream_hook_filter, backward_hook, "bwd")

        clean_tokens = clean_tokens.to(self.device)
        torch.cuda.empty_cache()
        value = metric(self.model(clean_tokens, return_type="logits"))
        value.backward()
        
        # we delete the activations difference tensor to free up memory
        self.model.zero_grad()
        del upstream_activations_difference
        gc.collect()
        torch.cuda.empty_cache()

        self.eap_scores = self.eap_scores.cpu()

        return self.eap_scores

    def top_edges(self, eap_scores=None, n=10, abs=True):
        if eap_scores is None:
            eap_scores = self.eap_scores

        # get indices of maximum values in 2d tensor
        if abs:
            top_scores, top_indices = torch.topk(eap_scores.flatten().abs(), k=n, dim=0)
        else:
            top_scores, top_indices = torch.topk(eap_scores.flatten(), k=n, dim=0)

        top_edges = []
        print("\nTop edges:")
        for i, (abs_score, index) in enumerate(zip(top_scores, top_indices)):
            upstream_node_idx, downstream_node_idx = np.unravel_index(index, eap_scores.shape)
            score = eap_scores[upstream_node_idx, downstream_node_idx]
            top_edges.append((self.upstream_nodes[upstream_node_idx], self.downstream_nodes[downstream_node_idx], score.item()))
            if i <= 10: 
                # we only print the first 10
                print(f"{score:.4f}\t{self.upstream_nodes[upstream_node_idx]} -> {self.downstream_nodes[downstream_node_idx]}")

        return top_edges

    def forward_with_patching(
        self,
        clean_tokens: Int[Tensor, "batch_size seq_len"],
        corrupted_tokens: Int[Tensor, "batch_size seq_len"],
        patching_edges: List[Tuple]
    ) -> Float[Tensor, "batch_size seq_len d_vocab"]:

        upstream_nodes = [edge[0] for edge in patching_edges]
        downstream_nodes = [edge[1] for edge in patching_edges]

        self.setup_edges(upstream_nodes, downstream_nodes)

        adj_matrix = torch.zeros((self.n_upstream_nodes, self.n_downstream_nodes), device=self.device, dtype=self.dtype)

        print(f"Number of upstream nodes is {self.n_upstream_nodes}")
        print(f"Number of downstream nodes is {self.n_downstream_nodes}")

        for upstream_node, downstream_node in zip(upstream_nodes, downstream_nodes): 
            upstream_node_index = self.upstream_node_index[upstream_node]
            downstream_node_index = self.downstream_node_index[downstream_node]

            adj_matrix[upstream_node_index, downstream_node_index] = 1.0

        seq_len = clean_tokens.shape[1]
        batch_size = clean_tokens.shape[0]
        assert seq_len == corrupted_tokens.shape[1], "Sequence length mismatch"
        assert batch_size == corrupted_tokens.shape[0], "Batch size mismatch"

        activations_diff_in_gb = batch_size * seq_len * self.n_upstream_nodes * self.d_model * self.dtype.itemsize / 2**30 
        memory_allocated_in_gb = torch.cuda.memory_allocated() / 2**30
        total_memory_in_gb = torch.cuda.get_device_properties(self.device).total_memory / 2**30
        print(f"Saving activation differences requires {activations_diff_in_gb:.2f} GB of memory")
        assert activations_diff_in_gb < total_memory_in_gb - memory_allocated_in_gb, "Not enough memory to store activation differences"

        self.model.reset_hooks()
        gc.collect()
        torch.cuda.empty_cache()

        upstream_activations_difference = torch.zeros((batch_size, seq_len, self.n_upstream_nodes, self.d_model), device=self.device, requires_grad=False)

        memory_allocated_in_gb = torch.cuda.memory_allocated() / 2**30
        print(f"Total memory allocated after creating activation differences tensor is {memory_allocated_in_gb:.2f} GB out of {total_memory_in_gb:.2f} GB")
        
        def corrupted_upstream_hook(activations, hook):
            hook_slice = self.upstream_hook_slice[hook.name]
            if activations.ndim == 3:
                upstream_activations_difference[:, :, hook_slice, :] = -activations.unsqueeze(-2)
            elif activations.ndim == 4:
                upstream_activations_difference[:, :, hook_slice, :] = -activations

        def clean_upstream_hook(activations, hook):
            hook_slice = self.upstream_hook_slice[hook.name]
            if activations.ndim == 3:
                upstream_activations_difference[:, :, hook_slice, :] += activations.unsqueeze(-2)
            elif activations.ndim == 4:
                upstream_activations_difference[:, :, hook_slice, :] += activations

        def clean_downstream_hook(activations, hook):
            hook_slice = self.downstream_hook_slice[hook.name]

            earlier_upstream_nodes_slice = self.get_slice_previous_upstream_nodes(hook)

            # the tensor 'patch_difference' represents the sum of all upstream activation differences that are connected to this downstream node
            patch_difference = einops.einsum(
                adj_matrix[earlier_upstream_nodes_slice, hook_slice],
                upstream_activations_difference[:, :, earlier_upstream_nodes_slice, :],
                "n_upstream n_downstream_at_hook, batch_size seq_len n_upstream d_model -> batch_size seq_len n_downstream_at_hook d_model"
            )

            # alternatively, it might be faster to
            # 1) gather the activation differences for every non-zero element in adj_matrix[:, hook_slice] and sum them
            # 2) use torch.sparse.mm to multiply the adj_matrix with the activation differences

            if activations.ndim == 3:
                assert patch_difference.shape[-2] == 1, "Number of downstream nodes should be 1 for this type of hook" 
                activations = activations - patch_difference.squeeze(-2)
            elif activations.ndim == 4:
                activations = activations - patch_difference
            
            return activations

        upstream_hook_filter = lambda name: name.endswith(tuple(self.upstream_hooks))
        downstream_hook_filter = lambda name: name.endswith(tuple(self.downstream_hooks))

        # we first perform a forward pass on the corrupted input 
        self.model.add_hook(upstream_hook_filter, corrupted_upstream_hook, "fwd")

        with torch.no_grad():
            corrupted_tokens = corrupted_tokens.to(self.device)
            self.model(corrupted_tokens, return_type=None)

        self.model.reset_hooks()
        torch.cuda.empty_cache()

        self.model.add_hook(upstream_hook_filter, clean_upstream_hook, "fwd")
        self.model.add_hook(downstream_hook_filter, clean_downstream_hook, "fwd")

        with torch.no_grad():
            clean_tokens = clean_tokens.to(self.device)
            logits = self.model(clean_tokens, return_type="logits")

        self.model.reset_hooks()
        del upstream_activations_difference
        torch.cuda.empty_cache()
        gc.collect()

        return logits 
