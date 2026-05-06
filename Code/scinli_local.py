from datasets import  load_dataset 
import pandas as pd
import time
import random
from vllm import LLM
from vllm.sampling_params import SamplingParams
import re
from huggingface_hub import login


with open("/home/tmalik6/LLMR/Code/hf_login.txt", "r") as f:
        token = f.read().strip()
    
login(token = token)
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_id2 = "meta-llama/Llama-3.1-8B-Instruct"
llm = LLM(
    model=model_id2,
    dtype="float16",              # or "float16" depending on your GPU
    tensor_parallel_size=1,        # >1 if you have multiple GPUs
    gpu_memory_utilization=0.9,
    max_model_len=4096
    )

tok = llm.get_tokenizer()

def input_dataset(data_name, version=""):
    if version == "":
        dataset = load_dataset(data_name)
    else:
        dataset = load_dataset(data_name, version)
    return dataset

def extract_ans(text):
    lines = text.strip().splitlines()
    
    # Iterate backwards to find the first non-empty line
    for line in reversed(lines):
        if line.strip():  # Non-empty after stripping spaces
            return line.strip()
    
    return None  # No non-empty lines found

def chat(system_prompt, user_prompt, n, seed, temperature=0.7, max_tokens=2048):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    sampling = SamplingParams(
        temperature=temperature,
        n=n,                        # multiple samples from the same prompt
        max_tokens=max_tokens,
        seed=seed
    )
    out = llm.generate(prompt, sampling, use_tqdm=False)[0].outputs
    completions = [o.text.strip() for o in out]

    return completions

def solve(sent1, sent2, label, sc = 1):
     # For tracking performance
    times = []
    
    # For storing answers
    full_l8 = [[] for _ in range(sc)]
    
    system = "You are a helpful assistant."
    
    # Process each question
    for i in range(len(sent1)):
        
        print(f'Question {i + 1} in progress...')
        
        user = (
            "Consider the following two sentences:\n"
            f"Sentence1 (premise): {sent1[i]}\n"
            f"Sentence2 (hypothesis): {sent2[i]}\n"
            "Based only on the information available in these two sentences, which of the following options is true?\n"
            "a. Sentence1 generalizes, specifies or has an equivalent meaning with Sentence2.\n"
            "b. Sentence1 presents the reason, cause, or condition for the result or conclusion made Sentence2.\n"
            "c. Sentence2 mentions a comparison, criticism, juxtaposition, or a limitation of something said in Sentence1.\n"
            "d. Sentence1 and Sentence2 are independent.\n"
            "Task: Determine the relation between Sentence 1 and Sentence 2.\n"
            "Options: entailment, neutral, contrasting, reasoning.\n"
            "Denote your chosen option at the end of your response in the following format:" 
            "\n Final Answer: ..."
        )
       
        time_start = time.time()
        full_sc = chat(system, user, n=sc, seed = 7,temperature= 0.5,max_tokens= 2048)        
        time_end = time.time()
        # print(full_sc)
        for idx, sample in enumerate(full_sc):
            full_l8[idx].append(sample)
            # print(sample)

        times.append(time_end - time_start)

    
    # Create DataFrame with results
    df_data = {
        "Sentence 1": sent1,
        "Sentence 2": sent2,
        "Correct Answer": label
    }
    for i in range(sc):
        df_data[f"Llama8B"] = full_l8[i]
    
    df_debate_full = pd.DataFrame(df_data)
    
    
    print("\n----- Average Response Times -----")
    print(f"{sum(times)/len(times):.2f}s")
    print()
    
    return df_debate_full


def main():
    

    dataset = input_dataset("tasksource/scinli")

    # df = pd.read_csv("/home/tmalik6/LLMR/Code/scinli/test.csv")
    sent1 = dataset['train']["sentence1"][2500:8000]
    sent2 = dataset['train']["sentence2"][2500:8000]
    labels = dataset['train']["label"][2500:8000]

    df = solve(sent1, sent2, labels, 1)
    df.to_csv('SciNLI_train_L8_25_to_8k.csv', index=True)

main()