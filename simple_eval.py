from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Load model using vLLM
model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
llm = LLM(model=model_name)

# Sampling settings
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=100,
)

# Load dataset
dataset = load_dataset("Sebasdi/art_math_test", split="train")

# Evaluation loop
correct = 0
total = 0

def build_prompt(question: str) -> str:
    return f"### Question:\n{question}\n" #### Answer:

def normalize(text: str) -> str:
    return text.strip().lower().rstrip(".")

print("Evaluating...")

for example in tqdm(dataset[:10]):
    prompt = build_prompt(example["question"])
    outputs = llm.generate([prompt], sampling_params)
    generated = outputs[0].outputs[0].text.strip()

    print("Answer:", example["answer"])
    print("Generated:", generated)
    print("\n")

    #pred = normalize(generated)
#     gold = normalize(example["answer"])

#     if pred in gold or gold in pred:  # simple overlap check
#         correct += 1
#     total += 1

# accuracy = correct / total * 100
# print(f"Accuracy: {accuracy:.2f}% on {total} samples")
