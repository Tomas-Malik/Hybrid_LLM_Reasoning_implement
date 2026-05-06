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
                temperature=0.5,
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


def solve(qs, ans, client):
     # For tracking performance
    times = []
    
    # For storing answers
    full_l8 = []
    
    system = "You are a helpful assistant."
    
    # Process each question
    for i in range(len(qs)):
        
        print(f'Question {i + 1} in progress...')
        user = "As an expert problem solver, solve step by step the following question and choose the right answer from the presented options.\n\n"
        user = user + f'Q: {qs[i]}\nA: Let\'s think step by step.\n\n'
        user = user + f"Indicate your final decision as the final answer as True or False on the final line of your output as follows: \n\n Final Answer: ..."
        user = user + f"If you're unsure, take a guess, but always return a True or False at the end."
    
       
        
        full_sc, x = groq_one_sample(client, user, system, temperature=0.5, max_tokens=2048,retries=3, backoff=2)
        
        full_l8.append(full_sc)
        

        

    
    # Create DataFrame with results
    df_data = {
        "Question": qs,
        "Correct Answer": ans,
        "l70": full_l8
    }
    
    
    
    df_debate_full = pd.DataFrame(df_data)
    
    
   
    
    return df_debate_full
def main():

    dataset_load = "ChilleD/StrategyQA"
    dataset=input_dataset(dataset_load)

    questions = dataset['train']['question']
    answers = dataset['train']['answer']

    with open("/home/tmalik6/LLMR/groq_api.txt") as file:
        api_k = file.read()

    client = Groq(api_key=api_k)

    

    df = solve(questions,answers, client)
    df.to_csv('SQA_train_l70.csv', index=True)

    

main()