# RL Fine-tuning for Positive Sentiment Generation

A hands-on exercise for learning Reinforcement Learning fine-tuning of Language Models using GRPO (Group Relative Policy Optimization).

## Overview

In this exercise, you will:
1. Implement reward shaping functions
2. Implement KL divergence regularization
3. Train GPT-2 to generate positive sentiment text
4. Compare different training configurations

## Prerequisites

- Python 3.8+
- GPU with 8GB VRAM
- Basic understanding of PyTorch, Transformers, and RL concepts

---

## Setup

### 1. Install Miniconda (if needed)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### 2. Create Environment

```bash
conda create -n sentiment python=3.10 -y
conda activate sentiment
```

### 3. Install PyTorch and Dependencies

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python rewards.py  # Should show NotImplementedError for student functions
```

---

## Project Structure

```
sentiment/
├── rewards.py            # Student exercises (implement TODOs here)
├── rewards_solution.py   # Solutions (instructor only)
├── reward_utils.py       # Reward infrastructure (provided)
├── sentiment.py          # Sentiment model (provided)
├── train.py              # Training script
├── data.py               # Prompt dataset
└── README.md
```

---

## Exercises

### Exercise 1: Reward Shaping

In `rewards.py`, implement `shaped_reward()` to transform raw sentiment scores.

Ideas to try:
- **Exponential**: Amplify differences from neutral sentiment
- **Length penalty**: Penalize very short/long responses
- **Repetition penalty**: Detect and penalize "great great great" outputs

### Exercise 2: KL Regularization

Implement `kl_penalty_forward()` and `kl_penalty_backward()` to prevent the model from drifting too far from the original GPT-2.

You receive pre-computed log probabilities for both the policy and reference models. Design penalties that discourage deviation.

---

## Training

### Basic Training

```bash
python train.py
```

### With Custom Reward Shaping

```bash
python train.py --reward_shaping shaped
```

### With KL Regularization

```bash
# TRL's built-in KL (efficient)
python train.py --beta 0.1

# Your custom KL implementation
python train.py --kl_type forward --kl_coef 0.1
```

### With Weights & Biases Logging

```bash
python train.py --wandb
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--reward_shaping` | linear, shaped | linear |
| `--kl_type` | none, forward, backward | none |
| `--kl_coef` | Custom KL strength | 0.1 |
| `--beta` | TRL's internal KL strength | 0.0 |
| `--hackable_reward` | Use exploitable word-counting reward | False |
| `--negate_reward` | Optimize for negative sentiment | False |

---

## Troubleshooting

**Out of Memory**: Add `--use_peft` for LoRA training

**Import Errors**: `pip install -r requirements.txt --upgrade`

---

## References

- [TRL Documentation](https://huggingface.co/docs/trl)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
