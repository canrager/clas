# Circuit-Level Activation Steering

We identify circuits in LLMs based on [Edge Attribution Patching](https://arxiv.org/abs/2310.10348), and [Automatic Circuit Discovery](https://arxiv.org/abs/2304.14997). We are currently supporting [TransformerLens](https://github.com/neelnanda-io/TransformerLens) models only. In the future, we may apply [Activation Steering](https://arxiv.org/abs/2308.10248) to the identified circuits.

## Current state
See [demo notebook](https://github.com/canrager/clas/blob/main/first_implementation_demo/EAP_Demo.ipynb). Oscar implemented EAP by having a very big tensor of activation differences between clean and corrupted activations that we write and read from. We calculate the EAP scores during the backward pass to save memory and only store the difference between positive and negative activations instead of storing both. 

## TODOs
- [ ] Positional EAP: Compute edge attribution scores across different positions (Answering questions like: "Does the token *bomb* at some middle position in the prompt have a high impact on the final position?")
- [ ] Graph plotting (use ACDC functionality or web-based approach?)
- [ ] Generalize Metric: Compute general metric from logits **or intermediate activations** (
- [ ] Activation steering: Abstract away the localization of hooks and positions to enable forward passes with
    1. Activation Steering (at arbitrary layers and sequence positions)
    2. Ablation (zero / mean)
    3. Custom Steering vector
- [ ] Write unit tests
- [ ] check poetry build requirements
