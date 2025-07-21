from datasets import load_dataset
from unsloth import FastLanguageModel   
from tqdm import tqdm
import re

# Load model using Unsloth
model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
model_name = "unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit"
model_name = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
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



def normalize(text: str) -> str:
    return text.strip().lower().rstrip(".")

print("Evaluating...")

for example in tqdm(dataset.select(range(100))):
    question = example["question"]
    messages = [
        {"role": "system", "content": "You are a clever agent that is asked to solve math questions. If you want to think before the answer do so in between the <think> and </think> tags. Provide your final answer in the format <answer>your answer</answer>"},
        {"role": "user", "content": f"""Question:\n{question}\n\n"""}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    outputs = model.generate(**inputs, max_new_tokens=250, use_cache=True)
    generated = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)[0]

    print("Answer:", example["answer"])
    print("Generated:", generated)


    extracted_answer = re.search(r'<answer>(.*?)</answer>', generated)
    if extracted_answer:
        pred = extracted_answer.group(1)
        print("extracted answer:", pred)
        print("\n")
    else:
        pred = ""

    if pred in example["answer"] or example["answer"] in pred:
        correct += 1
    total += 1
    # write question answer and extracted answer to file
    with open("results.txt", "a") as f:
        f.write(f"Question: {question}\n")
        f.write(f"Answer: {example['answer']}\n")
        f.write(f"Extracted answer: {pred}\n")
        f.write("\n")

accuracy = correct / total * 100
print(f"Accuracy: {accuracy:.2f}% on {total} samples")
