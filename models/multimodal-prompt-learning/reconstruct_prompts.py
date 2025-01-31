import torch
import os
from pathlib import Path
from clip import clip
import numpy as np
from typing import Dict, List, Tuple

class PromptReconstructor:
    def __init__(self, prompt_path: str, class_names: List[str]):
        """
        Initialize the prompt reconstructor with saved prompts and target classes.
        
        Args:
            prompt_path: Path to the saved learned prompts from MaPLe training
            class_names: List of raw class names (e.g., ["chair", "table", etc.])
        """
        # Load the saved prompt components
        self.learned_prompts = torch.load(prompt_path)
        self.class_names = class_names
        
        # Load the saved prompt components
        self.learned_prompts = torch.load(prompt_path)
        self.class_names = class_names
        
        # Load CLIP model with the same configuration as training
        design_details = {
            "trainer": 'MaPLePromptScene',
            "vision_depth": 0,
            "language_depth": 0, 
            "vision_ctx": 0,
            "language_ctx": 0,
            "maple_length": self.learned_prompts['n_ctx']  # Use the saved n_ctx value
        }
        
        # Get the model path and state dict like in MaPLe training
        backbone_name = "ViT-B/16"  # This should match your training configuration
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)
        
        try:
            # Try loading as JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            # Fall back to loading state dict
            state_dict = torch.load(model_path, map_location="cpu")
        
        # Build model with design details
        self.clip_model = clip.build_model(state_dict or model.state_dict(), design_details)
        self.clip_model = self.clip_model.to('cuda')
        
        # Verify we have the expected number of classes
        if len(class_names) != self.learned_prompts['token_prefix'].shape[0]:
            raise ValueError(
                f"Number of classes ({len(class_names)}) doesn't match "
                f"saved prompts ({self.learned_prompts['token_prefix'].shape[0]})"
            )

    def reconstruct_complete_prompts(self) -> torch.Tensor:
        """
        Reconstruct the complete prompts by combining learned components.
        
        Returns:
            Tensor of shape [num_classes, sequence_length, embedding_dim]
        """
        # Get components from saved state
        ctx = self.learned_prompts['ctx']                    # [num_ctx, dim]
        prefix = self.learned_prompts['token_prefix']        # [num_classes, 1, dim]
        suffix = self.learned_prompts['token_suffix']        # [num_classes, suffix_len, dim]
        compound_prompts = self.learned_prompts['compound_prompts_text']  # List of [num_ctx, dim]
        
        # Expand context vectors for each class
        batch_ctx = ctx.unsqueeze(0).expand(len(self.class_names), -1, -1)  # [num_classes, num_ctx, dim]
        
        # Combine in the sequence: [SOS] + [learned_ctx] + [class_name + EOS]
        complete_prompts = torch.cat([
            prefix,      # [num_classes, 1, dim]
            batch_ctx,   # [num_classes, num_ctx, dim]
            suffix       # [num_classes, suffix_len, dim]
        ], dim=1)       # Result: [num_classes, sequence_length, dim]
        
        return complete_prompts, compound_prompts

    def encode_prompts(self) -> torch.Tensor:
        """
        Encode the complete prompts through CLIP's text encoder.
        
        Returns:
            Tensor of shape [num_classes, clip_dim] containing the final text embeddings
        """
        complete_prompts, compound_prompts = self.reconstruct_complete_prompts()
        
        # Get the tokenized format of our class prompts
        tokenized_prompts = self.learned_prompts['tokenized_prompts']
        
        # Pass through CLIP's text encoder
        with torch.no_grad():
            # Use the same text encoder configuration as in training
            text_features = self.clip_model.encode_text(
                complete_prompts, 
                tokenized_prompts,
                compound_prompts
            )
            
            # Normalize the features as done in training
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        return text_features

    def save_encoded_prompts(self, save_path: str):
        """
        Save the encoded prompts along with class names for inference.
        
        Args:
            save_path: Where to save the encoded prompts
        """
        encoded_prompts = self.encode_prompts()
        
        # Save both the embeddings and their corresponding class names
        torch.save({
            'text_embeddings': encoded_prompts,
            'class_names': self.class_names
        }, save_path)
        
        print(f"Saved encoded prompts to {save_path}")
        print(f"Shape: {encoded_prompts.shape}")
        print(f"Number of classes: {len(self.class_names)}")

#from reconstruct_prompts import PromptReconstructor
#import torch
#
## Define paths
#prompt_path = "output/replica/MaPLePromptScene/vit_b16_c2_ep200_batch4_ctx4_depth12_cross_datasets_-1shots/seed1/learned_prompts/learned_prompts_final.#pt"
#output_path = "replica_prompts.pt"
#
## Define class names
#class_names = [
#    "basket", "bed", "bench", "bin", "blanket", "blinds", "book", "bottle", "box", "bowl", "camera", "cabinet", "candle", "chair", "clock", "cloth", "comforter", "cushion", "desk", "desk-organizer", "door", "indoor-plant", "lamp", "monitor", "nightstand", "panel", "picture", "pillar", "pillow", "pipe", "plant-stand", "plate", "pot", "sculpture", "shelf", "sofa", "stool", "switch", "table", "tablet", "tissue-paper", "tv-screen", "tv-stand", "vase", "vent", "wall-plug", "window", "rug"
#]
#
## Create the reconstructor
#reconstructor = PromptReconstructor(prompt_path, class_names)
#
## You can examine intermediate results if you want
#complete_prompts, compound_prompts = reconstructor.reconstruct_complete_prompts()
#print("Complete prompts shape:", complete_prompts.shape)
#print("Number of compound prompt layers:", len(compound_prompts))
#
## Get the encoded prompts
#encoded_prompts = reconstructor.encode_prompts()
#print("\nEncoded prompts shape:", encoded_prompts.shape)
#print("Encoded prompts statistics:")
#print(f"Mean: {encoded_prompts.mean().item():.6f}")
#print(f"Std: {encoded_prompts.std().item():.6f}")
#
## Save the final results
#reconstructor.save_encoded_prompts(output_path)