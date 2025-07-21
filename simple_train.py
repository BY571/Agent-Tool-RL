import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from tqdm import tqdm
import re
from unsloth import is_bfloat16_supported
import numpy as np
import wandb
import torch.nn.functional as F


# Load model with LoRA adapter using Unsloth
#model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
#model_name = "unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit"
model_name = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
lora_rank = 16
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    random_state = 3407,
)
FastLanguageModel.for_training(model)

# Define optimizer for LoRA parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Hyperparameters
num_epochs = 300
update_every_n = 10
gradient_accumulation_steps = update_every_n

# Load dataset
dataset = load_dataset("Sebasdi/art_math_test", split="train")

# Create a new results file
with open("results.txt", "w") as f:
    f.write("")

print(f"Training for {num_epochs} epochs...")

wandb.init(project="art-math", name="simple_train")

steps = 0
for epoch in range(num_epochs):
    print(f"\nStarting epoch {epoch + 1}/{num_epochs}")
    # Evaluation loop
    correct = 0
    total = 0
    log_probs = []
    rewards = []

    for idx, example in enumerate(tqdm(dataset.select(range(100)), desc=f"Epoch {epoch + 1}")):
        steps += 1
        question = example["question"]
        messages = [
                {"role": "system", "content": (
            "You are a clever math-solving assistant. "
            "You have access to a calculator tool to perform mathematical computations. "
            "Use the <tool>calculate: expression</tool> tag to invoke the calculator tool as <tool>calculate: 3*63</tool>. "
            "Ensure the expression is clear, follows standard mathematical notation, and includes necessary parentheses for clarity. "
            "If you need to reason through the problem before invoking the tool, do so within <think> and </think> tags. "
            "Provide your final answer in the format <answer>your answer</answer> like: <answer>42</answer>."),
                },
                {"role": "user", "content": f"""Question:\n{question}\n\n"""}
            ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]

        # First generation (may include tool call)
        model.train()
        with torch.enable_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
                use_cache=False,
                return_dict_in_generate=True,
                temperature=0.8,
            )
            answer_token = outputs.sequences[:, input_len:]
            generated = tokenizer.batch_decode(answer_token, skip_special_tokens=True)[0]

            # Extract log probabilities for the first generation
            inputs = torch.cat([inputs["input_ids"], answer_token], -1)
            output = model(
                input_ids=inputs,
                )
            logits = output.logits[:, :-1, :]  # (batch, seq_len - 1, vocab_size)
            first_log_probs = F.log_softmax(logits, dim=-1)  # (batch, seq_len - 1, vocab_size)
            shifted_labels = inputs[:, 1:]  # (batch, seq_len - 1)
            first_log_probs = first_log_probs.gather(dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)[:, input_len:]

        # Process tool calls
        second_log_probs = None
        tool_call = re.search(r'<tool>calculate:\s*(.*?)</tool>', generated)
        if tool_call:
            expr = tool_call.group(1)
            try:
                tool_result = str(eval(expr, {"__builtins__": None}, {}))
            except Exception as e:
                tool_result = f"[ERROR: {str(e)}]"
            print("Tool call detected:", expr)
            print("Tool result:", tool_result)

            # Add tool result and re-generate
            messages.append({"role": "assistant", "content": generated})
            messages.append({"role": "tool", "content": f"Calculator tool result: {tool_result}"})
            messages.append(
                {
                    "role": "user",
                    "content": "Please continue. Please provide your final answer in the format <answer>your answer</answer>."
                }
            )
            prompt2 = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([prompt2], return_tensors="pt").to("cuda")
            input_len = inputs["input_ids"].shape[1]

            # Second generation with gradient tracking
            with torch.enable_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    use_cache=False,
                    return_dict_in_generate=True,
                    temperature=0.8,
                )
                answer_token = outputs.sequences[:, input_len:]
                generated = tokenizer.batch_decode(answer_token, skip_special_tokens=True)[0]
                # Extract log probabilities for the second generation
                inputs = torch.cat([inputs["input_ids"], answer_token], -1)
                output = model(
                    input_ids=inputs,
                    )
                logits = output.logits[:, :-1, :]  # (batch, seq_len - 1, vocab_size)
                second_log_probs = F.log_softmax(logits, dim=-1)  # (batch, seq_len - 1, vocab_size)
                shifted_labels = inputs[:, 1:]  # (batch, seq_len - 1)
                second_log_probs = second_log_probs.gather(dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)[:, input_len:]

        print("Answer:", example["answer"])
        print("Generated:", generated)

        # Extract predicted answer
        extracted_answer = re.search(r'<answer>(.*?)</answer>', generated)
        if extracted_answer:
            pred = extracted_answer.group(1)
            print("Extracted answer:", pred)
            print("\n")
        else:
            pred = "NONE"

        # Compute reward
        reward = 1 if pred in example["answer"] or example["answer"] in pred else 0
        correct += reward
        if pred == "NONE":
            reward = -0.1
        total += 1
        rewards.append(reward)

        # Combine log probabilities (first and second generations)
        combined_log_probs = first_log_probs
        if second_log_probs is not None:
            combined_log_probs = torch.cat([first_log_probs, second_log_probs], dim=1)
        log_probs.append(combined_log_probs)

        # Write to results file
        with open("results.txt", "a") as f:
            f.write(f"Epoch {epoch + 1}, Example {idx + 1}:\n")
            f.write(f"Question: {question}\n")
            f.write(f"Answer: {example['answer']}\n")
            f.write(f"Extracted answer: {pred}\n")
            f.write(f"Reward: {reward}\n")
            f.write("\n")

        # Policy gradient update every n examples
        if (idx + 1) % update_every_n == 0:
            print(f"Updating adapter after {idx + 1} examples in epoch {epoch + 1}...")
            model.zero_grad()

            # Compute policy gradient loss
            policy_loss = 0
            for log_prob, r in zip(log_probs, rewards):
                policy_loss -= r * log_prob.sum() - np.mean(rewards)

            policy_loss /= update_every_n
            policy_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            log_probs = []
            rewards = []

            print(f"Policy loss: {policy_loss.item():.4f}")
            wandb.log({"policy_loss": policy_loss.item()}, step=steps)

    # Epoch accuracy
    accuracy = correct / total * 100
    print(f"Epoch {epoch + 1} Accuracy: {accuracy:.2f}% on {total} samples")
    wandb.log({"accuracy": accuracy, "epoch": epoch + 1}, step=steps)

    # Save the updated adapter after each epoch
    #model.save_pretrained(f"lora_adapter_updated_epoch_{epoch + 1}")

# Final overall accuracy
print(f"\nTraining completed for {num_epochs} epochs.")