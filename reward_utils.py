"""
Reward utility functions for RL sentiment fine-tuning.

This module provides infrastructure for computing rewards, including:
- Log probability computation for KL regularization
- Factory function for creating combined reward functions

Students should focus on rewards.py for the exercise implementations.
"""

import torch
from typing import Callable, Optional


def compute_log_probs(
    completions: list[str],
    prompts: list[str],
    model,
    tokenizer,
) -> list[float]:
    """
    Compute normalized log probabilities of completions under a model.
    
    This computes how "likely" each completion is according to the given model.
    Used for KL regularization to compare policy vs reference distributions.
    
    Args:
        completions: List of generated text completions
        prompts: List of prompts that generated these completions
        model: The model to compute log probs under (policy or reference)
        tokenizer: Tokenizer for the model
    
    Returns:
        List of normalized log probabilities (per-token average)
        - Values are negative (log probs)
        - Higher (less negative) = more likely under model
        - Lower (more negative) = less likely under model
    """
    device = next(model.parameters()).device
    
    # Concatenate prompts and completions
    full_texts = [p + c for p, c in zip(prompts, completions)]
    
    # Tokenize full sequences
    inputs = tokenizer(
        full_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get prompt lengths to know where completions start
    prompt_inputs = tokenizer(
        prompts,
        padding=False,
        truncation=True,
        max_length=512,
    )
    prompt_lengths = [len(ids) for ids in prompt_inputs["input_ids"]]
    
    # Forward pass through model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (batch, seq_len, vocab_size)
    
    # Compute log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)
    input_ids = inputs["input_ids"]
    
    # Compute average log prob for completion tokens only
    normalized_log_probs = []
    for i in range(len(completions)):
        start_idx = prompt_lengths[i]
        seq_len = int(inputs["attention_mask"][i].sum().item())
        
        # Sum log probs for completion tokens
        # logits[t] predicts token[t+1], so we look at positions start_idx to seq_len-2
        seq_log_prob = 0.0
        num_tokens = 0
        for t in range(start_idx, seq_len - 1):
            token_id = input_ids[i, t + 1].item()
            seq_log_prob += log_probs[i, t, token_id].item()
            num_tokens += 1
        
        # Normalize by number of tokens
        avg_log_prob = seq_log_prob / max(num_tokens, 1)
        normalized_log_probs.append(avg_log_prob)
    
    return normalized_log_probs


def make_reward_function(
    shaping: str = "linear",
    kl_type: str = "none",
    kl_coef: float = 0.1,
    policy_model=None,
    ref_model=None,
    tokenizer=None,
    base_scorer: Optional[Callable] = None,
    negate: bool = False,
    reward_module=None,
):
    """
    Factory function to create a combined reward function.
    
    This creates a reward function that:
    1. Computes base reward using the scorer (with optional shaping)
    2. Adds KL regularization penalty (if enabled)
    3. Optionally negates the final reward (for negative sentiment optimization)
    
    Args:
        shaping: Type of reward shaping ("linear" or "exponential")
        kl_type: Type of KL regularization ("none", "forward", "backward")
        kl_coef: Coefficient for KL term
        policy_model: Current policy model (required if kl_type != "none")
        ref_model: Reference model (required if kl_type != "none")
        tokenizer: Tokenizer (required if kl_type != "none")
        base_scorer: Optional custom scorer function(texts) -> [0,1] scores.
                     If None, uses sentiment_reward from reward_module.
        negate: If True, negate the final reward (optimize for negative sentiment)
        reward_module: Module containing reward functions (rewards or rewards_solution)
    
    Returns:
        A reward function compatible with TRL's GRPOTrainer
    """
    if kl_type != "none" and (policy_model is None or ref_model is None or tokenizer is None):
        raise ValueError(f"policy_model, ref_model, and tokenizer required for kl_type={kl_type}")
    
    if reward_module is None:
        # Default to importing rewards_solution
        import rewards_solution as reward_module
    
    # Use provided scorer or default to sentiment_reward
    scorer = base_scorer if base_scorer is not None else reward_module.sentiment_reward
    
    def reward_fn(completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
        # Step 1: Compute base reward from scorer
        raw_scores = scorer(completions)
        
        # Step 2: Apply shaping
        if shaping == "linear":
            base_rewards = list(raw_scores)
        elif shaping == "shaped":
            base_rewards = reward_module.shaped_reward(raw_scores, completions)
        else:
            raise ValueError(f"Unknown shaping: {shaping}")
        
        # Step 3: Add KL regularization if enabled
        if kl_type != "none" and prompts is not None:
            # Compute log probs for both models (infrastructure - not part of student exercise)
            log_probs_policy = compute_log_probs(completions, prompts, policy_model, tokenizer)
            log_probs_ref = compute_log_probs(completions, prompts, ref_model, tokenizer)
            
            # Call student's KL function with pre-computed log probs
            if kl_type == "forward":
                kl_terms = reward_module.kl_penalty_forward(
                    log_probs_policy, log_probs_ref, kl_coef
                )
            elif kl_type == "backward":
                kl_terms = reward_module.kl_penalty_backward(
                    log_probs_policy, log_probs_ref, kl_coef
                )
            else:
                raise ValueError(f"Unknown kl_type: {kl_type}")
            
            # Combine: total_reward = base_reward + kl_term
            base_rewards = [base + kl for base, kl in zip(base_rewards, kl_terms)]
        
        # Step 4: Negate if optimizing for negative sentiment
        if negate:
            base_rewards = [-r for r in base_rewards]
        
        return base_rewards
    
    return reward_fn
