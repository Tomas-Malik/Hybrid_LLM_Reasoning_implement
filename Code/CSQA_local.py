from datasets import load_dataset 
import pandas as pd
from vllm import LLM
from vllm.sampling_params import SamplingParams
import time
import re
from huggingface_hub import login

with open("/home/tmalik6/LLMR/Code/hf_login.txt", "r") as f:
        token = f.read().strip()
    
login(token = token)
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

def extract_ans(text):
    lines = text.strip().splitlines()
    
    # Iterate backwards to find the first non-empty line
    for line in reversed(lines):
        if line.strip():  # Non-empty after stripping spaces
            return line.strip()
    
    return None  # No non-empty lines found
        
def self_consistency_chat(system_prompt, user_prompt, n, seed, temperature=0.5, max_tokens=2048):
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

def preprocess_options(options):
    otp = []
    for i, j in zip(options["label"], options["text"]):
        x = f"{i}. {j}"
        otp.append(x)
    return otp

def solve_sc(qs, options, ans, sc = 1):
    # For tracking performance
    times = []
    
    # For storing answers
    full_l8 = [[] for _ in range(sc)]
    
    system = "You are a helpful assistant."
    
    # Process each question
    for i in range(len(qs)):
        
        print(f'Question {i + 1} in progress...')
        options_curr = preprocess_options(options[i])
        user = "As an expert problem solver, solve step by step the following question and choose the right answer from the presented options.\n\n"
        user = user + f'Q: {qs[i]}\nA: Let\'s think step by step.\n\n'
        user = user + f"The options are: {options_curr}. Indicate the the latter and the word of the option you're choosing as the final answer in the following format: Final Answer: 'Letter'. 'Word'"
    
       
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
        "Options": options,
        "Correct Answer": ans,
    }
    for i in range(sc):
        df_data[f"Llama8B"] = full_l8[i]
    
    df_debate_full = pd.DataFrame(df_data)
    
    
    # print("\n----- Average Response Times -----")
    # print(f"{sum(times)/len(times):.2f}s")
    # print()
    
    return df_debate_full


    
        
        


def main():

    dataset_name="CSQA"



    dataset_dict = {
        "gsm8k": "openai/gsm8k", # 7.47k training size, 1.32k test size
        "MATHold": "lighteval/MATH", #outdated?
        "MATH": "xDAN2099/lighteval-MATH",
        "MATH500":  "HuggingFaceH4/MATH-500",
        "CSQA": "tau/commonsense_qa"
    }

    
    dataset_load = dataset_dict.get(dataset_name)
    dataset=input_dataset(dataset_load)


    questions = dataset['train']['question']
    options = dataset['train']['choices']
    answers = dataset['train']['answerKey']

    df = solve_sc(questions, options, answers, 1)
    df.to_csv('CSQA_llama8B_train.csv', index=True)

    

main()