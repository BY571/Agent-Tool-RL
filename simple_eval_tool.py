from datasets import load_dataset
from unsloth import FastLanguageModel   
from tqdm import tqdm
import re

# Load model using Unsloth
model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
#model_name = "unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit"
#model_name = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# Sampling settings
# Unsloth uses the transformers pipeline, which takes arguments directly.

# Load dataset
dataset = load_dataset("Sebasdi/art_math_test", split="train")

# Evaluation loop
correct = 0
total = 0

# create a new results file:
with open("results.txt", "w") as f:
    f.write("")



print("Evaluating...")

for example in tqdm(dataset.select(range(100))):
    question = example["question"]
    messages = [
        {"role": "system", "content": (
    "You are a clever math-solving assistant. "
    "You have access to a calculator tool to perform mathematical computations. "
    "Use the <tool>calculate: expression</tool> tag to invoke the calculator tool as <tool>calculate: 3*63</tool>. "
    "Ensure the expression is clear, follows standard mathematical notation, and includes necessary parentheses for clarity. "
    "If you need to reason through the problem before invoking the tool, do so within <think> and </think> tags. "
    "Provide your final answer in the format <answer>your answer</answer>."),
        },
        {"role": "user", "content": f"""Question:\n{question}\n\n"""}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    outputs = model.generate(**inputs, max_new_tokens=250, use_cache=True)
    generated = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)[0]

    tool_call = re.search(r'<tool>calculate:\s*(.*?)</tool>', generated)
    if tool_call:
        expr = tool_call.group(1)
        try:
            tool_result = str(eval(expr, {"__builtins__": None}, {}))
        except Exception as e:
            tool_result = f"[ERROR: {str(e)}]"
        print("Tool call detected:", expr)
        print("Tool result:", tool_result)

        # Add tool result as assistant message and re-generate final answer
        messages.append({"role": "assistant", "content": generated})
        messages.append({"role": "tool", "content": f"Calculator tool result: {tool_result}"})
        messages.append({"role": "user", "content": "Please continue. Please provide your final answer in the format <answer>your answer</answer>."})
        
        prompt2 = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt2], return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]
        outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
        generated = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)[0]


    print("Answer:", example["answer"])
    print("Generated:", generated)


    extracted_answer = re.search(r'<answer>(.*?)</answer>', generated)
    if extracted_answer:
        pred = extracted_answer.group(1)
        print("extracted answer:", pred)
        print("\n")
    else:
        pred = "NONE"

    if pred in example["answer"] or example["answer"] in pred:
        reward = 1
    else:
        reward = 0
    correct += reward
    total += 1
    # write question answer and extracted answer to file
    # create a fresh file each time

    with open("results.txt", "a") as f:
        f.write(f"Question: {question}\n")
        f.write(f"Answer: {example['answer']}\n")
        f.write(f"Extracted answer: {pred}\n")
        f.write(f"Reward: {reward}\n")
        f.write("\n")

accuracy = correct / total * 100
print(f"Accuracy: {accuracy:.2f}% on {total} samples")
