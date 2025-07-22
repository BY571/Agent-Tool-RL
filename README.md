# Simple Agent + Tool + RL Example

This repository explores the use of tools and reinforcement learning in an agent to solve math problems.
Its main purpose is in understanding how to train a SLM to use tools with RL.  

## Setup

```bash
conda env create -f environment.yml
conda activate art
```

## Scripts

- **simple_eval.py:** Simple evaluation script to test baseline agent  without tools.
- **simple_eval_tool.py:** Simple evaluation script to test agent with tools.
- **simple_train.py:** Simple training script to train agent with tools.



# Eval results 

## Baseline

| Model                          | Dataset | Accuracy (3 seeds)| Tools |
| :----------------------------- | :------ | :------- | :---- |
| Qwen2.5-3B-Instruct-bnb-4bit | math    |  90%, 94%, 91%      | none  |
| Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit | math    |  46%, 60%, 58%       | none  |
| Qwen2.5-0.5B-Instruct-bnb-4bit | math    |  12%, 12%, 15%      | none  |


## Tool

| Model                          | Dataset | Accuracy (3 seeds)| Tools |
| :----------------------------- | :------ | :------- | :---- |
| Qwen2.5-3B-Instruct-bnb-4bit | math    |  97%, 91%, 94%      | yes |
| Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit | math    |  38%, 41%, 41%      | yes |
| Qwen2.5-0.5B-Instruct-bnb-4bit | math    |  4%, 12%, 12%      | yes |

## Tool + RL

| Model                          | Dataset | Accuracy (3 seeds)| Tools |
| :----------------------------- | :------ | :------- | :---- |
| Qwen2.5-3B-Instruct-bnb-4bit | math    |  100%, 100%, 100%      | yes |
| Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit | math    |  100%, 100%, 100%      | yes |
| Qwen2.5-0.5B-Instruct-bnb-4bit | math    |  100%, 100%, 95%      | yes |

