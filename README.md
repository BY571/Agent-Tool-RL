# ART-Agentic-Reasoning-Tools

## Setup

```bash
conda env create -f environment.yml
conda activate art
```


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
| Qwen2.5-3B-Instruct-bnb-4bit | math    |  ?      | yes |
| Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit | math    |  100%, 100%, 100%      | yes |
| Qwen2.5-0.5B-Instruct-bnb-4bit | math    |  100%, 100%, 95%      | yes |

### TODO:
- [X] Setup environment  
- [X] Setup eval script to test baseline agent and how to use tools with prompts
- [ ] 
