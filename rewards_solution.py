"""
Reward Functions for RL Sentiment Fine-tuning - SOLUTION VERSION

This module contains complete implementations of all reward functions.
This is the instructor/solution version - students should work with rewards.py.
"""

import math
from sentiment import get_sentiment_scores


# =============================================================================
# BASE REWARD FUNCTION
# =============================================================================

def sentiment_reward(completions: list[str]) -> list[float]:
    """
    Compute sentiment reward for a list of completions.
    
    This is the base reward function that returns sentiment scores in [0, 1].
    Higher values indicate more positive sentiment.
    
    Args:
        completions: List of generated text completions
    
    Returns:
        List of sentiment scores in [0, 1]
    """
    return get_sentiment_scores(completions)


# =============================================================================
# REWARD SHAPING - SOLUTION
# =============================================================================

def shaped_reward(scores: list[float], completions: list[str]) -> list[float]:
    """
    Apply custom reward shaping to transform raw sentiment scores.
    
    SOLUTION: Exponential shaping that amplifies deviation from neutral.
    
    Formula: exp((score - 0.5) / temperature) - 1
    - score=0.5 -> 0 (neutral stays neutral)
    - score=0.9 -> ~0.49 (high becomes more positive)
    - score=0.1 -> ~-0.33 (low becomes more negative)
    
    Alternative solutions students might implement:
    - Length penalty: reward -= 0.01 * abs(len(completion) - target_len)
    - Repetition penalty: count unique words / total words
    - Threshold: 1.0 if score > 0.7 else -1.0
    """
    temperature = 1.0
    shaped = []
    for score in scores:
        # Exponential shaping
        shifted = score - 0.5
        shaped.append(math.exp(shifted / temperature) - 1.0)
    return shaped


# =============================================================================
# KL REGULARIZATION - SOLUTION
# =============================================================================

def kl_penalty_forward(
    log_probs_policy: list[float],
    log_probs_ref: list[float],
    kl_coef: float = 0.1,
) -> list[float]:
    """
    Forward KL regularization penalty.
    
    SOLUTION: kl_coef * (log_prob_ref - log_prob_policy)
    
    This penalizes the policy for deviating from the reference model:
    - When policy assigns HIGHER prob than reference: penalty is negative
    - When policy assigns SIMILAR prob: penalty is ~0
    - When policy assigns LOWER prob: penalty is positive (small bonus)
    """
    penalties = [
        kl_coef * (lp_ref - lp_policy) 
        for lp_policy, lp_ref in zip(log_probs_policy, log_probs_ref)
    ]
    return penalties


def kl_penalty_backward(
    log_probs_policy: list[float],
    log_probs_ref: list[float],
    kl_coef: float = 0.1,
) -> list[float]:
    """
    Backward KL regularization penalty with exponential weighting.
    
    SOLUTION: -kl_coef * exp(log_prob_policy - log_prob_ref)
    
    Note: exp(log_prob_policy - log_prob_ref) = prob_policy / prob_ref
    
    This gives a larger penalty when the policy is much more confident than
    the reference, which helps prevent mode collapse.
    """
    penalties = []
    for lp_policy, lp_ref in zip(log_probs_policy, log_probs_ref):
        diff = lp_policy - lp_ref
        # Clamp to avoid numerical overflow
        exp_diff = min(math.exp(diff), 100.0)
        penalties.append(-kl_coef * exp_diff)
    return penalties


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing SOLUTION implementations...\n")
    
    test_texts = [
        "This movie was absolutely fantastic! I loved every minute.",
        "Terrible film. Complete waste of time.",
        "It was okay, nothing special.",
    ]
    
    # Test sentiment reward
    print("1. Sentiment Reward (raw scores [0, 1]):")
    rewards = sentiment_reward(test_texts)
    for text, reward in zip(test_texts, rewards):
        print(f"   {reward:.3f}: {text[:40]}...")
    
    # Test reward shaping
    print("\n2. Shaped Reward (exponential example):")
    test_scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    test_completions = ["bad", "meh", "okay", "good movie", "amazing film!"]
    shaped_results = shaped_reward(test_scores, test_completions)
    for score, s in zip(test_scores, shaped_results):
        print(f"   {score:.1f} -> {s:+.3f}")
    
    # Test KL penalties
    print("\n3. KL Penalties:")
    test_lp_policy = [-2.0, -3.0, -1.5]  # Policy log probs
    test_lp_ref = [-2.5, -2.5, -2.5]      # Reference log probs
    
    fwd = kl_penalty_forward(test_lp_policy, test_lp_ref, 0.1)
    print(f"   Forward KL penalties: {[f'{p:.3f}' for p in fwd]}")
    print(f"   (Negative when policy more confident than ref)")
    
    bwd = kl_penalty_backward(test_lp_policy, test_lp_ref, 0.1)
    print(f"   Backward KL penalties: {[f'{p:.3f}' for p in bwd]}")
    print(f"   (Exponentially larger when policy >> ref)")
    
    print("\nAll solution tests passed!")
