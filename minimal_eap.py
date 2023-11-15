#%% Imports
import sys
sys.path.append('..')

import torch as t
import einops
import plotly.express as px

from jaxtyping import Float
from torch import Tensor

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from acdc.greaterthan.utils import get_all_greaterthan_things
from tl_utils import get_3_caches

device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
print(f'Device: {device}')

#%% Get transformer model running

model = HookedTransformer.from_pretrained(
    'gpt2-small',
    center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device=device,
)
model.set_use_hook_mlp_in(True)
model.set_use_split_qkv_input(True)
model.set_use_attn_result(True)

#%% Get clean and corrupting datasets and task specific metric
BATCH_SIZE = 50
things = get_all_greaterthan_things(
    num_examples=BATCH_SIZE, metric_name="greaterthan", device=device
)
greaterthan_metric = things.validation_metric
clean_ds = things.validation_data # clean data x_i
corr_ds = things.validation_patch_data # corrupted data x_i'

print("\nClean dataset samples")
for stage_cnt in range(5):
    print(model.tokenizer.decode(clean_ds[stage_cnt]))

print("\nReference dataset samples")
for stage_cnt in range(5):
    print(model.tokenizer.decode(corr_ds[stage_cnt]))

#%% Run the model on a dataset sample to verify the setup worked
next_token_logits = model(clean_ds[3])[-1, -1]
next_token_str = model.tokenizer.decode(next_token_logits.argmax())
print(f"prompt: {model.tokenizer.decode(clean_ds[3])}")
print(f"next token: {next_token_str}")

# %% Define Hook filters for upstream and downstream nodes
# Upstream nodes in {Embeddings ("blocks.0.hook_resid_pre"), Attn_heads ("result"), MLPs ("mlp_out")}
# Downstream nodes in {Attn_heads ("input") , MLPs ("mlp_in"), resid_final ("blocks.11.hook_resid_post")}
# Necessary Transformerlens flags: model.set_use_hook_mlp_in(True), model.set_use_split_qkv_input(True), model.set_use_attn_result(True)
upstream_hook_names = ("blocks.0.hook_resid_pre", "hook_result", "hook_mlp_out")
downstream_hook_names = ("hook_q_input","hook_k_input", "hook_v_input", "hook_mlp_in", "blocks.11.hook_resid_post")

# %% Get the required caches for calculating EAP scores
# (2 forward passes on clean and corr ds, backward pass on clean ds)

clean_cache, corrupted_cache, clean_grad_cache = get_3_caches(
    model,
    clean_ds,
    corr_ds,
    greaterthan_metric,
    mode="edge",
    upstream_hook_names=upstream_hook_names,
    downstream_hook_names=downstream_hook_names
)

# %% Compute matrix holding all attribution scores
# edge_attribution_score = (upstream_corr - upstream_clean) * downstream_grad_clean
SEQUENCE_LENGTH = clean_ds.shape[1]

N_TOTAL_UPSTREAM_NODES = 1 + model.cfg.n_layers * (model.cfg.n_heads + 1)
N_TOTAL_DOWNSTREAM_NODES = 1 + model.cfg.n_layers * (3*model.cfg.n_heads + 1)

# Get (upstream_corr - upstream_clean) as matrix
N_UPSTREAM_NODES = model.cfg.n_heads
upstream_cache_clean = t.zeros((
    N_TOTAL_UPSTREAM_NODES,
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    model.cfg.d_model
))
upstream_cache_corr = t.zeros_like(upstream_cache_clean)

upstream_names = []
upstream_levels = t.zeros(N_TOTAL_UPSTREAM_NODES)
idx = 0
for stage_cnt, name in enumerate(clean_cache.keys()): # stage_cnt relevant for keeping track which upstream-downstream mairs can be connected
    if name.endswith("result"): # layer of attn heads
        act_clean = einops.rearrange(clean_cache[name], "b s nh dm -> nh b s dm")
        act_corr = einops.rearrange(corrupted_cache[name], "b s nh dm -> nh b s dm")
        upstream_cache_clean[idx:idx+model.cfg.n_heads] = act_clean
        upstream_cache_corr[idx:idx+model.cfg.n_heads] = act_corr
        upstream_levels[idx:idx+model.cfg.n_heads] = stage_cnt
        idx += model.cfg.n_heads
        for i in range(model.cfg.n_heads):
            upstream_names.append([name, i])
    else:
        upstream_cache_clean[idx] = clean_cache[name]
        upstream_cache_corr[idx] = corrupted_cache[name]
        upstream_levels[idx] = stage_cnt
        idx += 1
        upstream_names.append([name, -1])

upstream_diff = upstream_cache_corr - upstream_cache_clean

#%% Get downstream_grad as matrix
N_DOWNSTREAM_NODES = model.cfg.n_heads * 3 # q, k, v separate
downstream_grad_cache_clean = t.zeros((
    N_TOTAL_DOWNSTREAM_NODES,
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    model.cfg.d_model
))

downstream_names = []
downstream_levels = t.zeros(N_TOTAL_DOWNSTREAM_NODES)
stage_cnt = 0
idx = 0
names = reversed(list(clean_grad_cache.keys()))
for name in names:
    if name.endswith("hook_q_input"): # do all q k v hooks of that layer simultaneously, as it is the same stage
        q_name = name
        k_name = name[:-7] + "k_input"
        v_name = name[:-7] + "v_input"
        q_act = einops.rearrange(clean_grad_cache[q_name], "b s nh dm -> nh b s dm")
        k_act = einops.rearrange(clean_grad_cache[k_name], "b s nh dm -> nh b s dm")
        v_act = einops.rearrange(clean_grad_cache[v_name], "b s nh dm -> nh b s dm")

        downstream_grad_cache_clean[idx: idx+model.cfg.n_heads] = q_act
        downstream_levels[idx: idx+model.cfg.n_heads] = stage_cnt
        idx += model.cfg.n_heads
        for i in range(model.cfg.n_heads):
            downstream_names.append([q_name, i])
        
        downstream_grad_cache_clean[idx: idx+model.cfg.n_heads] = k_act
        downstream_levels[idx: idx+model.cfg.n_heads] = stage_cnt
        idx += model.cfg.n_heads
        for i in range(model.cfg.n_heads):
            downstream_names.append([k_name, i])

        downstream_grad_cache_clean[idx: idx+model.cfg.n_heads] = v_act
        downstream_levels[idx: idx+model.cfg.n_heads] = stage_cnt
        idx += model.cfg.n_heads
        for i in range(model.cfg.n_heads):
            downstream_names.append([v_name, i])

    elif name.endswith(("hook_k_input", "hook_v_input")):
        continue
    else:
        downstream_grad_cache_clean[idx] = clean_grad_cache[name]
        downstream_levels[idx] = stage_cnt
        idx += 1
        downstream_names.append([name, -1])
    stage_cnt += 1

#%% Calculate the cartesian product of stage, node for upstream and downstream
eap_scores = einops.einsum(
    upstream_diff, 
    downstream_grad_cache_clean,
    "up_nodes batch seq d_model, down_nodes batch seq d_model -> up_nodes down_nodes"
)

#%% Make explicit only upstream -> downstream (not downstream -> upstream is important)
upstream_level_matrix = einops.repeat(upstream_levels, "up_nodes -> up_nodes down_nodes", down_nodes=N_TOTAL_DOWNSTREAM_NODES)
downstream_level_matrix = einops.repeat(downstream_levels, "down_nodes -> up_nodes down_nodes", up_nodes=N_TOTAL_UPSTREAM_NODES)
mask = upstream_level_matrix > downstream_level_matrix
eap_scores = eap_scores.masked_fill(mask, value=t.nan)

px.imshow(
    eap_scores,
    x = [name+str(idx) if idx>=0 else name for name, idx in downstream_names ],
    y = [name+str(idx) if idx>=0 else name for name, idx in upstream_names],
    labels = dict(x="downstream node", y="upstream node", color="EAP score"),
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0
)














# %% Generate a dictionary for the most important nodes, sorted by absolute values
# maybe add an edge class for this?
TOP_K = 50


topk_eap_scores = []
for u, un in enumerate(upstream_names):
    for d, dn in enumerate(downstream_names):
        topk_eap_scores.append([un, dn, eap_scores[u,d].abs().item()])
topk_eap_scores = sorted(topk_eap_scores, key=lambda x: x[2], reverse=True)[:TOP_K]
topk_eap_scores[:5]

#%% Sort topk edges by layer, drop eap scores
topk_dict = dict()
for edge in topk_eap_scores:
    d_name = edge[1][0]
    if d_name not in topk_dict.keys():
        topk_dict[d_name] = [edge[:2]]
    else:
        topk_dict[d_name].append(edge[:2])
#%%

def steering_hook_attn(
    attn_pattern: Float[Tensor, "batch seq head_idx d_model"],
    hook: str
):
    """
    Global variables needed:
    - topk_dict
    - clean_cache
    - corr_cache
    """
    # get down-head, up-head indices pairs (i, j)
    for [up_name, up_idx], [down_name, down_idx] in topk_dict[hook.name]:
        assert hook.name == down_name, f"{hook.name=}{down_name=}"
        # get steering vector
        steering_vector = retrieve_steering_vector(clean_cache, corrupted_cache, up_name, up_idx)
        attn_pattern[:, :, down_idx, :] += steering_vector # alpha tuneable parameter
    return attn_pattern

def steering_hook_mlp_resid(
    attn_pattern: Float[Tensor, "batch seq d_model"],
    hook: str
):
    for [up_name, up_idx], [down_name, down_idx] in topk_dict[hook.name]:
        assert hook.name == down_name
        assert down_idx == -1 # this layer should contain one node only
        # get steering vector
        steering_vector = retrieve_steering_vector(clean_cache, corrupted_cache, up_name, up_idx)
        attn_pattern += steering_vector # alpha tuneable parameter
    return

def retrieve_steering_vector(clean_cache, corr_cache, up_hook_name, up_hook_index):
    print(f'{up_hook_name=}, {up_hook_index=}')
    if up_hook_index >= 0: # attn_head
        clean_cache_vec = clean_cache[up_hook_name][:, :, up_hook_index, :]
        corr_cache_vec = corr_cache[up_hook_name][:, :, up_hook_index, :]
    else:
        clean_cache_vec = clean_cache[up_hook_name]
        corr_cache_vec = corr_cache[up_hook_name]
        
    clean_steering_vector = einops.reduce(
        clean_cache_vec, 
        "batch seq d_model -> seq d_model", 
        "mean")
    corr_steering_vector = einops.reduce(
        corr_cache_vec, 
        "batch seq d_model -> seq d_model", 
        "mean")

    return clean_steering_vector - corr_steering_vector


#%%
#%% Inference
prompt = clean_ds[3][:5]

for i in range(10):
    next_token_logits = model(prompt)[-1, -1]
    next_token_id = next_token_logits.argmax()
    prompt = t.hstack((prompt, next_token_id))
# %%
model.tokenizer.decode(prompt)

#%% Run model with hooks
model.reset_hooks()
for d_name in topk_dict.keys():
    if d_name[-5:] == "input":
        model.add_hook(d_name, steering_hook_attn, "fwd")
    else:
        model.add_hook(d_name, steering_hook_mlp_resid, "fwd")

#%% Inference
prompt = clean_ds[3]

for i in range(10):
    next_token_logits = model(prompt)[-1, -1]
    next_token_id = next_token_logits.argmax()
    prompt = t.hstack((prompt, next_token_id))
# %%
model.tokenizer.decode(prompt)

# %%
