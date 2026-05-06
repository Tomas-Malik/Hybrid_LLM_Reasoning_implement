from datasets import  load_dataset 
import pandas as pd
from groq import Groq
import time
import random

def input_dataset(data_name, version = ""):
    if version == "":
        dataset = load_dataset(data_name)
    else:
        dataset = load_dataset(data_name,version)
    return dataset

def extract_ans(text):
    lines = text.strip().splitlines()
    
    # Iterate backwards to find the first non-empty line
    for line in reversed(lines):
        if line.strip():  # Non-empty after stripping spaces
            return line.strip()
    
    return None  # No non-empty lines found
        

def groq_one_sample(client, user, system, temperature=0.5, max_tokens=2048,
                    retries=3, backoff=2):
    for _ in range(retries):
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    ],
                temperature=temperature,
                max_completion_tokens=max_tokens
            )
            return [resp.choices[0].message.content], None
        except Exception as e:
            msg = str(e)
            if "503" in msg or "Service Unavailable" in msg:
                time.sleep(backoff); backoff *= 2
                continue
            return None, msg
    return None, "failed after retries"


def solve(qs, ans, evidence, client, sc = 1):
     # For tracking performance
    times = []
    
    # For storing answers
    full_l8 = [[] for _ in range(sc)]
    
    system = "You are a helpful assistant."
    
    # Process each question
    for i in range(len(qs)):
        
        print(f'Question {i + 1} in progress...')
        user = f"Here is evidence relevant to the target question: {evidence[i]}.\n\n"
        user = user + "As an expert problem solver, solve step by step the following question and choose the right answer from the presented options.\n\n"
        user = user + f'Q: {qs[i]}\nA: Let\'s think step by step.\n\n'
        user = user + f"Indicate your final decision as the final answer as True or False on the final line of your output as follows: \n\n Final Answer: ..."
        user = user + f"If you're unsure, take a guess, but always return a True or False at the end."
    
       
        time_start = time.time()
        full_sc, x = groq_one_sample(client, user, system, temperature=0.5, max_tokens=2048,retries=3, backoff=2)
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
        df_data[f"Llama70B SC ({i+1})"] = full_l8[i]
    
    df_debate_full = pd.DataFrame(df_data)
    
    
    print("\n----- Average Response Times -----")
    print(f"{sum(times)/len(times):.2f}s")
    print()
    
    return df_debate_full
def main():

    dataset_load = "ChilleD/StrategyQA"
    dataset=input_dataset(dataset_load)

    questions = dataset['train']['question']
    answers = dataset['train']['answer']
    evidence = dataset['train']['facts']

    with open("/home/tmalik6/LLMR/groq_api.txt") as file:
        api_k = file.read()

    client = Groq(api_key=api_k)

    

    df = solve(questions,answers, evidence, client, 1)
    df.to_csv('StrategyQA_l70_we_train.csv', index=True)

    

main()