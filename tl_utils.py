from typing import Tuple, Literal, Optional
from transformer_lens import ActivationCache


def get_3_caches(
    model,
    clean_input,
    corrupted_input,
    metric,
    mode: Literal["node", "edge"] = "node",
    upstream_hook_names: Optional[Tuple] = None,
    downstream_hook_names: Optional[Tuple] = None,
):
    """
    This function is written by @Aaquib111 and originally part of https://github.com/Aaquib111/acdcpp/blob/main/utils/prune_utils.py
    """

    # default hook names
    if not upstream_hook_names:
        upstream_hook_names = (
            "hook_result",
            "hook_mlp_out",
            "blocks.0.hook_resid_pre",
            "hook_q",
            "hook_k",
            "hook_v",
        )
    if not downstream_hook_names:
        if model.cfg.attn_only:
            downstream_hook_names = (
                "hook_q_input",
                "hook_k_input",
                "hook_v_input",
                f"blocks.{model.cfg.n_layers-1}.hook_resid_post",
            )
        else:
            downstream_hook_names = (
                "hook_mlp_in",
                "hook_q_input",
                "hook_k_input",
                "hook_v_input",
                f"blocks.{model.cfg.n_layers-1}.hook_resid_post",
            )

    # cache the activations and gradients of the clean inputs
    model.reset_hooks()
    clean_cache = {}

    def forward_cache_hook(act, hook):
        clean_cache[hook.name] = act.detach()

    edge_acdcpp_outgoing_filter = lambda name: name.endswith(upstream_hook_names)
    model.add_hook(edge_acdcpp_outgoing_filter, forward_cache_hook, "fwd")

    clean_grad_cache = {}

    def backward_cache_hook(act, hook):
        clean_grad_cache[hook.name] = act.detach()

    edge_acdcpp_back_filter = lambda name: name.endswith(downstream_hook_names)
    model.add_hook(edge_acdcpp_back_filter, backward_cache_hook, "bwd")
    value = metric(model(clean_input))
    value.backward()

    # cache the activations of the corrupted inputs
    model.reset_hooks()
    corrupted_cache = {}

    def forward_corrupted_cache_hook(act, hook):
        corrupted_cache[hook.name] = act.detach()

    model.add_hook(edge_acdcpp_outgoing_filter, forward_corrupted_cache_hook, "fwd")
    model(corrupted_input)
    model.reset_hooks()

    clean_cache = ActivationCache(clean_cache, model)
    corrupted_cache = ActivationCache(corrupted_cache, model)
    clean_grad_cache = ActivationCache(clean_grad_cache, model)
    return clean_cache, corrupted_cache, clean_grad_cache
