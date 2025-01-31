
import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from tqdm import tqdm
import numpy as np

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    
    design_details = {"trainer": 'MaPLePromptScene',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.MAPLE_PROMPT_SCENE.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model



class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        # self.dtype = next(clip_model.parameters()).dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        # x = prompts + self.positional_embedding.type(self.dtype)
        x = prompts + self.positional_embedding.to(prompts.device).type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MAPLE_PROMPT_SCENE.N_CTX
        ctx_init = cfg.TRAINER.MAPLE_PROMPT_SCENE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        self.compound_prompts_depth = cfg.TRAINER.MAPLE_PROMPT_SCENE.PROMPT_DEPTH # max=12, but will create 11 such shared prompts
        assert self.compound_prompts_depth >= 1, "PROMPT_DEPTH should be >= 1"

        # print("\n=== Initializing MultiModalPromptLearner ===")
        # print(f"Number of classes: {n_cls}")
        # print(f"Context length: {n_ctx}")
        # print(f"Initial context: {ctx_init}")
        

        if ctx_init and (n_ctx) <= 4:
            # print("\nTokenization check:")
            
            ctx_init = ctx_init.replace("_", " ")
            # print(f"Processed context: {ctx_init}")
            demo_prompt = ctx_init.replace("{label}", classnames[0])
            # print(f"Sample complete prompt: {demo_prompt}")

            tokens = clip.tokenize(demo_prompt)
            # print(f"Tokenized prompt shape: {tokens.shape}")

            token_ids = tokens[0].tolist()

            name_token_idx = 1  # Start after SOS token
            for idx, token_id in enumerate(token_ids[1:], 1):
                if token_id != 0 and token_id != 49407:  # Skip padding and EOS
                    name_token_idx = idx
                    break
            # print(f"Class name token position: {name_token_idx}")


            with torch.no_grad():
                embedding = clip_model.token_embedding(tokens).type(dtype)
                # print(f"Initial embedding shape: {embedding.shape}")
                # print(f"Embedding dtype: {embedding.dtype}")
            
                start_idx = max(1, name_token_idx - n_ctx//2)
                end_idx = start_idx + n_ctx
                ctx_vectors = embedding[0, start_idx:end_idx, :]
                # print(f"Context vectors shape: {ctx_vectors.shape}")

                num_actual_tokens = sum(1 for t in token_ids if t != 0 and t != 49407)
                # print(f"Number of actual text tokens: {num_actual_tokens}")

            prompt_prefix = ctx_init

        else:
            # random initialization
            # print("\nRandom initialization used")
            
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            # print(f"Random context vectors shape: {ctx_vectors.shape}")
            
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        # Shallow layers
        self.ctx = nn.Parameter(ctx_vectors)

        # Deeper layers
        self.compound_prompts_text = nn.ParameterList([
            nn.Parameter(torch.empty(n_ctx, ctx_dim))
            for _ in range(self.compound_prompts_depth - 1)
        ])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)

        # print('MaPLe design: Multi-modal Prompt Learning')
        # print(f'Initial context: "{prompt_prefix}"')
        # print(f"Number of MaPLe context words (tokens): {n_ctx}")

        # Process classnames
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix.replace("{label}", name) for name in classnames]

        print("whole prompts")
        print(prompts)

        # Tokenize prompts
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        # print(f"Tokenized prompts shape: {tokenized_prompts.shape}")
        # print("\nDetailed tokenization analysis:")
        for i, p in enumerate(prompts[:3]):
            tokens = clip.tokenize(p)
            token_ids = tokens[0].tolist()
            meaningful_tokens = [t for t in token_ids if t != 0 and t != 49407]  # Remove padding and EOS
            # print(f"Prompt {i}: '{p}'")
            # print(f"  Raw token shape: {tokens.shape}")
            # print(f"  Meaningful tokens: {len(meaningful_tokens)}")
            # print(f"  Token IDs: {meaningful_tokens}")
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # Save special tokens (SOS, EOS + padding)
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # EOS + padding

        # Save other stuff
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            # print(f"Label values: {label}")
            prefix = prefix[label]
            suffix = suffix[label]

        # Number of actual text tokens in the suffix
        n_text_tokens = (suffix.shape[1] + 1) // 2
        
        # Split tokens into two parts
        ctx_before = self.n_ctx // 2

        # print("\nPrompt construction debug:")
        # print(f"Context shape: {ctx.shape}")
        # print(f"Prefix shape: {prefix.shape}")
        # print(f"Suffix shape: {suffix.shape}")
        # print(f"Number of text tokens in suffix: {n_text_tokens}")

        prompts = torch.cat(
            [
                prefix,  # [SOS] (dim0, 1, dim)
                ctx[:, :ctx_before],  # [first half of learnable tokens] (dim0, n_ctx, dim)
                suffix[:, :1],  # [class name] (dim0, *, dim)
                ctx[:, ctx_before:self.n_ctx],  # [second half of learnable tokens] (dim0, n_ctx, dim)
                suffix[:, 1:],  # [remaining tokens incl EOS] (dim0, *, dim)
            ],
            dim=1,
        )
        # print("\nPrompt construction verification:")
        # print(f"Prefix (SOS) size: {prefix.shape}")
        # print(f"First ctx half size: {ctx[:, :ctx_before].shape}")
        # print(f"Class token size: {suffix[:, :1].shape}")
        # print(f"Second ctx half size: {ctx[:, ctx_before:self.n_ctx].shape}")
        # print(f"Remaining suffix size: {suffix[:, 1:].shape}")
        # print(f"Total prompt size: {prompts.shape}")

        expected_length = 77  # CLIP's expected sequence length
        actual_length = prompts.shape[1]
        assert actual_length == expected_length, f"Prompt length {actual_length} doesn't match CLIP's expected length {expected_length}"

        return prompts

    def forward(self):
        # print("\n=== MultiModalPromptLearner Forward Pass ===")
        ctx = self.ctx

        # print(f"Context vectors dtype after conversion: {self.ctx.data.dtype}")

        if ctx.dim() == 2:
            # print(f"Expanding context from shape {ctx.shape}")
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            # print(f"to shape {ctx.shape}")

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)
        # print(f"Constructed prompts shape: {prompts.shape}")

        # Before returning, need to transform
        # prompts to 768 for the visual side
        # print("\nPrompt Projections:")
        # visual_deep_prompts = []
        # for index, layer in enumerate(self.compound_prompt_projections):
        #     projection = layer(self.compound_prompts_text[index])
        #     print (f"Layer {index} projection shape: {projection.shape}")
        #     visual_deep_prompts.append(projection)
        # 
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        
        # print(f"\nCtx dtype before projection: {self.ctx.dtype}")
        # projected_ctx = self.proj(self.ctx)
        # print(f"Ctx shape after projection: {projected_ctx.shape}")
        # print(f"Ctx dtype after projection: {projected_ctx.dtype}")

        return prompts, None, self.compound_prompts_text, None   # pass here original, as for visual 768 is required


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        # self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        # self.dtype = next(clip_model.parameters()).dtype

    def forward(self, point_features, label=None):
        # print("\n=== Starting CustomCLIP Forward Pass ===")
        # print(f"Point features shape: {point_features.shape}")
        # print(f"Point features dtype: {point_features.dtype}")

        tokenized_prompts = self.tokenized_prompts
        # print(f"Tokenized prompts shape: {tokenized_prompts.shape}")

        logit_scale = self.logit_scale.exp()
        # print(f"Logit scale: {logit_scale.item()}")

        prompts, _, deep_compound_prompts_text, _ = self.prompt_learner()
        # print(f"Prompts shape: {prompts.shape}")
        # print(f"Prompts dtype: {prompts.dtype}")
        
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        # print(f"Text features shape: {text_features.shape}")
        # print(f"Text features dtype: {text_features.dtype}")

        point_features = point_features / point_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # print(f"Normalized point features range: [{point_features.min().item()}, {point_features.max().item()}]")
        # print(f"Normalized text features range: [{text_features.min().item()}, {text_features.max().item()}]")
        
        # print(f"point_features.dtype", point_features.dtype)
        # print(f"point_features.dtype", text_features.dtype)
        
        logits = logit_scale * point_features @ text_features.t()  # calculate similarity
        # print(f"Logits shape: {logits.shape}")
        # print(f"Logits range: [{logits.min().item()}, {logits.max().item()}]")
    

        if self.prompt_learner.training:
            loss = F.cross_entropy(logits, label)
            # print(f"Training loss: {loss.item()}")
            
            return loss
        
        return logits


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class MaPLePromptScene(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MAPLE_PROMPT_SCENE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MAPLE_PROMPT_SCENE.PREC == "fp32" or cfg.TRAINER.MAPLE_PROMPT_SCENE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptSceneLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MAPLE_PROMPT_SCENE.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)


    def check_gradients(self, model):
        """
        Analyzes the gradients of trainable parameters to help debug training issues.
        
        Args:
            model: The model whose gradients we want to check
            
        Prints detailed information about:
        - Which parameters are being updated
        - The range and distribution of gradient values 
        - Any potential issues like vanishing/exploding gradients
        """
        print("\n=== Gradient Analysis ===")
        
        # Track if we find any concerning gradient patterns
        has_vanishing_grads = False
        has_exploding_grads = False

        for name, param in model.named_parameters():
            # Only check parameters we're actually training
            if param.requires_grad:
                if param.grad is not None:
                    # Calculate key statistics about the gradients
                    grad_min = param.grad.min().item()
                    grad_max = param.grad.max().item()
                    grad_mean = param.grad.mean().item()
                    grad_std = param.grad.std().item()
                    
                    print(f"\nParameter: {name}")
                    print(f"  Shape: {param.grad.shape}")
                    print(f"  Gradient range: [{grad_min:.6f}, {grad_max:.6f}]")
                    print(f"  Gradient mean: {grad_mean:.6f}")
                    print(f"  Gradient std: {grad_std:.6f}")

                    # Check for potential gradient issues
                    if abs(grad_mean) < 1e-7:
                        has_vanishing_grads = True
                        print("  WARNING: Possibly vanishing gradients")
                    if abs(grad_mean) > 1e2:
                        has_exploding_grads = True
                        print("  WARNING: Possibly exploding gradients")
                    
                    # Check for NaN values
                    if torch.isnan(param.grad).any():
                        print("  ERROR: NaN values detected in gradients!")
                else:
                    print(f"\nParameter: {name}")
                    print("  No gradients computed for this parameter")
        
        # Print summary of findings
        print("\n=== Gradient Check Summary ===")
        if has_vanishing_grads:
            print("- Warning: Some parameters have very small gradients")
        if has_exploding_grads:
            print("- Warning: Some parameters have very large gradients")
        if not (has_vanishing_grads or has_exploding_grads):
            print("- All gradient values appear to be in a reasonable range")


    def forward_backward(self, batch):
        point_feature, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        # print("\n=== Starting Training Step ===")
        # print(f"Batch size: {point_feature.shape[0]}")

        # Record initial parameter states and print every 100 batches
        """
        if self.batch_idx % 100 == 0:  
            print("\nParameters before forward pass:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"{name} statistics:")
                    print(f"  Mean: {param.data.mean().item():.6f}")
                    print(f"  Std: {param.data.std().item():.6f}")
        """
        prec = self.cfg.TRAINER.MAPLE_PROMPT_SCENE.PREC
        if prec == "amp":
            with autocast():
                loss = model(point_feature, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            
            # Check gradients before optimization step
            # if self.batch_idx % 100 == 0:
            #     self.check_gradients(model)
            
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(point_feature, label)
            optim.zero_grad()
            loss.backward()

            # Check gradients before optimization step
            # if self.batch_idx % 100 == 0:
            #     self.check_gradients(model)

            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["feature"]
        label = batch["label"]
        # print(f"Input features shape: {input.shape}")
        # print(f"Input features dtype before conversion: {input.dtype}")
        # print(f"Label shape: {label.shape}")
        # print(f"Unique labels in batch: {torch.unique(label).tolist()}")

        input = input.to(torch.float16)
        # print(f"Input features dtype after conversion: {input.dtype}")

        input = input.to(self.device)
        label = label.to(self.device)
        # print(f"Device: {input.device}")
        # print(f"NaN values in features: {torch.isnan(input).any().item()}")
        # print(f"Feature value range: [{input.min().item()}, {input.max().item()}]")

        return input, label
    
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        predictions_whole = []
        scenename_whole = []
        idx_instance_whole = []
        confidences_whole = []
        
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            print("input:", input)
            # print("output:", output)
            # print("classname:", batch['classname'])
            # print("scenename:", batch['scenename'])
            # print("idx_instance:", batch['idx_instance'])
            
            softmax_output = F.softmax(output, dim=1)
            
            predictions = torch.argmax(softmax_output, dim=1)
            confidences = torch.max(softmax_output, dim=1)[0]
            # print("\n=== Prediction Analysis ===")
            # print(f"Raw logits shape: {output.shape}")
            # print(f"Logits range: [{output.min().item():.4f}, {output.max().item():.4f}]")
            # print(f"Softmax probabilities range: [{softmax_output.min().item():.4f}, {softmax_output.max().item():.4f}]")
            # print(f"Predicted classes: {predictions.tolist()}")
            # print(f"True labels: {label.tolist()}")
            # print(f"Accuracy in batch: {(predictions == label).float().mean():.4f}")
            
            """
            # Print top-3 predictions for first few examples in batch
            n_examples = min(3, len(label))
            for i in range(n_examples):
                top_probs, top_idxs = torch.topk(softmax_output[i], k=3)
                print(f"\nExample {i}:")
                print(f"True label: {label[i].item()}")
                print("Top 3 predictions:")
                for prob, idx in zip(top_probs, top_idxs):
                    print(f"  Class {idx.item()}: {prob.item():.4f}")
            """
            predictions_whole.append(predictions)
            scenename_whole.extend(batch['scenename'])
            idx_instance_whole.append(batch['idx_instance'])
            confidences_whole.append(confidences)

            self.evaluator.process(output, label)

        predictions_whole = torch.cat(predictions_whole)
        idx_instance_whole = torch.cat(idx_instance_whole)
        confidences_whole = torch.cat(confidences_whole)
        
        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        if split == "test":
            self.write_results(predictions_whole, confidences_whole, scenename_whole, idx_instance_whole)
        return list(results.values())[0]
    def write_results(self, preds, confidences, scenenames, idx_instances):
        scene_name_set = set(scenenames)
        scenenames = np.array(scenenames)
        for scene_name in scene_name_set:
            num_instance = 100 # !!!ADJUST
            pred_labels = np.zeros(num_instance, dtype=int)
            pred_confidence = np.zeros(num_instance, dtype=float)
            for i in range(num_instance):
                if i in idx_instances[scenenames == scene_name]:
                    pred = preds[scenenames == scene_name]
                    idx_instance = idx_instances[scenenames == scene_name]
                    confidence = confidences[scenenames == scene_name]
                    pred_labels[i] = pred[idx_instance == i]
                    pred_confidence[i] = confidence[idx_instance == i]
            np.savetxt(f"results/pred_class_{scene_name}.txt", pred_labels, fmt='%d')
            np.savetxt(f"results/pred_confidence_{scene_name}.txt", pred_confidence, fmt='%f')
        
            
    def parse_batch_test(self, batch):
        input = batch["feature"].to(torch.float16)
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
            
