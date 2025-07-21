# ART-Agentic-Reasoning-Tools

## Setup

```bash
conda env create -f environment.yml
conda activate art
```


# Eval results 

## Baseline

| Model                          | Dataset | Accuracy (2 seeds)| Tools |
| :----------------------------- | :------ | :------- | :---- |
| Qwen2.5-3B-Instruct-bnb-4bit | math    |  ~ 94%      | none  |
| Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit | math    |  ~ 80%      | none  |
| Qwen2.5-0.5B-Instruct-bnb-4bit | math    |  ~ 68%      | none  |



### TODO:
- [X] Setup environment  
- [X] Setup eval script to test baseline agent and how to use tools with prompts
- [ ] 
