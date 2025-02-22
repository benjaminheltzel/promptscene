import torch
import os
import numpy as np
from pathlib import Path

def inspect_prompt_file(prompt_path):
    """
    Loads and analyzes a saved prompt file, providing detailed information about
    each component and their properties.
    
    Args:
        prompt_path: Path to the saved prompt file (.pt format)
    """
    print(f"\nInspecting prompts from: {prompt_path}")
    
    # Load the saved prompts
    prompts = torch.load(prompt_path)
    
    # Basic component information
    print("\n=== Component Shapes ===")
    for key, value in prompts.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            print(f"{key}: {len(value)} tensors of shape {value[0].shape}")
        else:
            print(f"{key}: {type(value)}")
    
    # Detailed analysis of learned context vectors
    print("\n=== Context Vector Analysis ===")
    ctx = prompts['ctx']
    print("Statistics:")
    print(f"  Mean: {ctx.mean().item():.6f}")
    print(f"  Std:  {ctx.std().item():.6f}")
    print(f"  Min:  {ctx.min().item():.6f}")
    print(f"  Max:  {ctx.max().item():.6f}")
    
    # Analyze the deeper layer prompts
    if 'compound_prompts_text' in prompts:
        print("\n=== Compound Prompts Analysis ===")
        for idx, prompt in enumerate(prompts['compound_prompts_text']):
            print(f"\nLayer {idx}:")
            print(f"  Shape: {prompt.shape}")
            print(f"  Mean:  {prompt.mean().item():.6f}")
            print(f"  Std:   {prompt.std().item():.6f}")
            print(f"  Range: [{prompt.min().item():.6f}, {prompt.max().item():.6f}]")

    # Token structure analysis
    print("\n=== Token Structure ===")
    if 'token_prefix' in prompts and 'token_suffix' in prompts:
        print(f"Prefix tokens per class: {prompts['token_prefix'].shape[1]}")
        print(f"Suffix tokens per class: {prompts['token_suffix'].shape[1]}")
    
    return prompts  # Return the loaded prompts for further analysis if needed


# import torch
# from inspect_prompts import inspect_prompt_file
# 
# prompts = inspect_prompt_file("path/to/learned_prompts_final.pt")
# 
# ctx = prompts['ctx']
# correlation = torch.corrcoef(ctx)
# print("\nContext vector correlations:")
# print(correlation)