# Circuit-Level Activation Steering (CLAS)

CLAS finds the most important edges in Transformer models. It is a lightweight version of the [ACDC framework](https://github.com/ArthurConmy/Automatic-Circuit-Discovery). Further features may include a researcher-friendly interface for activation steering.

## Objectives
1. Automate Edge Attribution Patching
  - User input:
    1. clean_ds: context + feature
    2. corr_ds: context w/o feature
    3. HuggingFace / TransformerLens model with hooks
    4. list of upstream nodes and downstream nodes
    5. generalized metric
   
  - Repo functionality:
    1. Compute general metric from logits **or intermediate activations**
    2. Compute edge attribution scores across different positions
  
   
## Further Objectives to implement either here or in the ACDC & Transformerlens libraries
1. Object with all relevant edges (already part of ACDC)
2. Plotting edges above threshold (web-based only, could be added to ACDC)
3. Activation Steering (could be added to TransformerLens)
  - User input:
    1. List of edges
    2. Activation vector

  - Repo functionality: Abstract away the localization of hooks and positions to enable forward passes with
    1. Pruining (zero or mean ablation)
    2. Activation Steering (at arbitrary layers and sequence positions)

