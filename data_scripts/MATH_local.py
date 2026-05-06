from datasets import  load_dataset 
import pandas as pd
from vllm import LLM
from vllm.sampling_params import SamplingParams
import time
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
    max_model_len=2500
    )

tok = llm.get_tokenizer()

def input_dataset(data_name, version=""):
    if version == "":
        dataset = load_dataset(data_name)
    else:
        dataset = load_dataset(data_name, version)
    return dataset

def input_dataset(data_name, version = ""):
    if version == "":
        dataset = load_dataset(data_name, trust_remote_code=True)
    else:
        dataset = load_dataset(data_name,version, trust_remote_code=True)
    return dataset

def extract_math_ans(true_ans):
    init = len(true_ans) - 1 
    targ_len = len("\\boxed")
    final_ans = ""
    
    # Find the position of "\boxed"
    while (init - targ_len) >= 0:
        if true_ans[init - targ_len: init] == "\\boxed":
            break
        init -= 1
    if init - targ_len < 0:  # If "\boxed" is not found
        return "NA"

    # Locate the opening "{"
    start = init
    while start < len(true_ans) and true_ans[start] != "{":
        start += 1
    if start == len(true_ans):
        return "NA"  # No opening brace found

    # Extract the content inside \boxed{...}
    start += 1  # Move past '{'
    brackets = 1
    for i in range(start, len(true_ans)):
        if true_ans[i] == "{":
            brackets += 1
        elif true_ans[i] == "}":
            brackets -= 1
            if brackets == 0:
                return final_ans  # Stop when the last closing '}' is found
        final_ans += true_ans[i]

    return "NA"  # If it reaches here, something went wrong

        
def self_consistency_chat(system_prompt, user_prompt, n, seed, temperature=0.7, max_tokens=2048):
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

def solve(qs, ans, sc=1):
    times = []
    
    # For storing answers
    full_l8 = [[] for _ in range(sc)]
    
    
    system = "You are a helpful assistant that interacts entirely in LaTeX code. All responses should be formatted in LaTeX, including explanations, equations, and text."
    
    for i in range(len(qs)):
        

        print(f'Question {i + 1} in progress...')
        
        user = "As an expert problem solver, solve step by step the following mathematical question.\n\n"
        user = user + f'Q: {qs[i]}\nA: Let\'s think step by step.\n\n'
        user = user + "Use the following template for structing your answer: \n\n Step 1: \n\n Step 2: \n\n ... \n\n Last Step: \n\n Final answer: \\boxed{...}"
        
       
        time_start = time.time()
        full_sc = self_consistency_chat(system, user, n=sc, seed = 7,temperature= 0.5,max_tokens= 2048)        
        time_end = time.time()
        for idx, sample in enumerate(full_sc):
            full_l8[idx].append(sample)
            # print(sample)

        times.append(time_end - time_start)
        
    
    # Create DataFrame with results
    df_data = {
        "Question": qs,
        "Correct Answer": ans,
    }
    for i in range(sc):
        df_data[f"Llama8B SC ({i+1})"] = full_l8[i]
    
    df_debate_full = pd.DataFrame(df_data)
    
    
    print("\n----- Average Response Times -----")
    print(f"{sum(times)/len(times):.2f}s")
    print()
    
    return df_debate_full

def main():

    
    dataset_name="MATH"

    dataset_dict = {
        "gsm8k": "openai/gsm8k", # 7.47k training size, 1.32k test size
        "MATHold": "lighteval/MATH", #outdated?
        "MATH": "xDAN2099/lighteval-MATH",
        "MATH500":  "HuggingFaceH4/MATH-500"
    }

    dataset_load = dataset_dict.get(dataset_name)
    dataset=input_dataset(dataset_load)

    questions = dataset['train']['problem']
    answers = dataset['train']['solution']
    
    p = questions
    a = answers
    
    df = solve(p, a, 1)
    df.to_csv("MATH_train_l8.csv", index=True)

main()