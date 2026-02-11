"""
Reward Functions for RL Sentiment Fine-tuning - STUDENT VERSION

This module contains reward function implementations for the RL exercise.
Students should implement the functions marked with TODO.

Functions to implement:
1. shaped_reward() - Apply reward shaping to rewards
2. kl_penalty_forward() - Forward KL divergence penalty
3. kl_penalty_backward() - Backward KL divergence penalty

The base sentiment_reward() is provided as a working example.
"""

from sentiment import get_sentiment_scores


# =============================================================================
# BASE REWARD FUNCTION (Provided - working example)
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
# KL REGULARIZATION (Exercise 1)
# =============================================================================

def kl_penalty_forward(
    log_probs_policy: list[float],
    log_probs_ref: list[float],
    kl_coef: float = 0.1,
) -> list[float]:
    """
    Forward KL regularization penalty.
    
    Args:
        log_probs_policy: List of avg log probs under current policy (one per completion, averaged over tokens)
        log_probs_ref: List of avg log probs under reference model (one per completion, averaged over tokens)
        kl_coef: Coefficient controlling regularization strength
    
    Returns:
        List of penalty values to ADD to the base reward
    
    TODO: Implement this function
    """
    # =========================================================================
    # YOUR CODE HERE
    # =========================================================================
    raise NotImplementedError(
        "Exercise 1a: Implement forward KL penalty"
    )
    # =========================================================================
    # END YOUR CODE
    # =========================================================================


def kl_penalty_backward(
    log_probs_policy: list[float],
    log_probs_ref: list[float],
    kl_coef: float = 0.1,
) -> list[float]:
    """
    Backward KL regularization penalty with stronger penalization of deviations.

    Args:
        log_probs_policy: List of avg log probs under current policy (one per completion, averaged over tokens)
        log_probs_ref: List of avg log probs under reference model (one per completion, averaged over tokens)
        kl_coef: Coefficient controlling regularization strength
    
    Returns:
        List of penalty values to ADD to the base reward
    
    TODO: Implement this function
    """
    # =========================================================================
    # YOUR CODE HERE
    # =========================================================================
    raise NotImplementedError(
        "Exercise 1b: Implement backward KL penalty"
    )
    # =========================================================================
    # END YOUR CODE
    # =========================================================================


# =============================================================================
# REWARD SHAPING (Exercise 2)
# =============================================================================

def shaped_reward(scores: list[float], completions: list[str]) -> list[float]:
    """
    Apply custom reward shaping to transform raw sentiment scores.
    
    Reward shaping modifies the raw reward signal to change learning dynamics.
    This is your chance to experiment with different shaping strategies.
    
    Args:
        scores: Raw sentiment scores in [0, 1] from the sentiment model
        completions: The generated text completions (useful for extracting length, word repetition, etc.)
    
    Returns:
        List of shaped reward values
    
    Potential ideas: numeric transformation (e.g. exponential, polynomial, log); penalize or encourage long responses;
                     penalize word repetitions; or any idea you think might help.
    
    TODO: Implement your chosen shaping strategy
    """
    # =========================================================================
    # YOUR CODE HERE
    # =========================================================================
    raise NotImplementedError(
        "Exercise 2: Implement reward shaping"
    )
    # =========================================================================
    # END YOUR CODE
    # =========================================================================


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing reward functions module...\n")
    
    test_texts = [
        "This movie was absolutely fantastic! I loved every minute.",
        "Terrible film. Complete waste of time.",
        "It was okay, nothing special.",
    ]
    
    # Test sentiment reward (provided implementation)
    print("1. Sentiment Reward (raw scores [0, 1]):")
    rewards = sentiment_reward(test_texts)
    for text, reward in zip(test_texts, rewards):
        print(f"   {reward:.3f}: {text[:40]}...")
    
    # Test reward shaping (student implementation)
    print("\n2. Reward Shaping (student exercise):")
    test_scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    test_completions = ["bad", "meh", "okay", "good movie", "amazing film!"]
    try:
        shaped = shaped_reward(test_scores, test_completions)
        for score, completion, s in zip(test_scores, test_completions, shaped):
            print(f"   {score:.1f} '{completion}' -> {s:.3f}")
    except NotImplementedError:
        print(f"   Not implemented yet")
    
    # Test KL penalties (student implementation)
    print("\n3. KL Penalties (student exercise):")
    test_lp_policy = [-2.0, -3.0, -1.5]  # Example log probs
    test_lp_ref = [-2.5, -2.5, -2.5]
    try:
        fwd = kl_penalty_forward(test_lp_policy, test_lp_ref, 0.1)
        print(f"   Forward KL: {fwd}")
    except NotImplementedError:
        print(f"   Forward KL: Not implemented yet")
    try:
        bwd = kl_penalty_backward(test_lp_policy, test_lp_ref, 0.1)
        print(f"   Backward KL: {bwd}")
    except NotImplementedError:
        print(f"   Backward KL: Not implemented yet")
    
    print("\nDone!")
