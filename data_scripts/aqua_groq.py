from datasets import load_dataset 
import pandas as pd
import os
from groq import Groq

import re

def input_dataset(data_name, version=""):
    if version == "":
        dataset = load_dataset(data_name, trust_remote_code=True)
    else:
        dataset = load_dataset(data_name, version, trust_remote_code=True)
    return dataset



def solve(qs, ans, client, model_name):
    
    # For storing answers
    full_answers = []
    
    
    
    failed_target_extract = 0
    
    system = "You are a helpful assistant."
    
    # Process each question
    for i in range(len(qs)):
        print(f'Question {i + 1} in progress...')
        
        # Extract target answer
        target_ans = (ans[i])
        
        if target_ans == "NA":
            target_ans = "NA_target"
            print("****")
            print(f"This is the failed extract ans (i+1):{i+1}")
            print(f"this is full text ans: {ans[i]}")
            print("Question skipped")
            print("****")
            failed_target_extract += 1
            
        
        
        user = (
           f""" Solve the following multiple-choice math problem.

            Question: {qs[i]}
        
            Think step by step, then give ONLY the final answer line in this exact format:
            Final Answer: LETTER"""

        )
        
        # user_baseline = "As an expert problem solver, solve step by step the following mathematical question.\n\n"
        # user_baseline += f'Q: {qs[i]}\nA: Let\'s think step by step.\n\n'
        # user_baseline += "Indicate the the latter and the word of the option you're choosing as the final answer in the following format: Final Answer: 'Letter'. 'number'"
        
        chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user}
        ],
        model="llama-3.3-70b-versatile",
        temperature = 0.5,
        seed = 42,
        max_tokens = 4000
        )
        
        
        text_baseline = chat_completion.choices[0].message.content
            

        
        full_answers.append(text_baseline)
        
        
        # Check if baseline is correct
        

            
        
    
    # Create DataFrame with results
    
    
    # Create DataFrame with full answers
    df_full = pd.DataFrame({
        "Question": qs,
        "Correct Answer": ans,
        f"{model_name} Full": full_answers
    })
    

    return df_full

def main():
    dataset_name = "aqua"
    model_name = "Gemma_9B"
    dataset_dict = {
        "gsm8k": "openai/gsm8k", # 7.47k training size, 1.32k test size
        "GSM": "apple/GSM-Symbolic", # 5k test set for P1, 2.5k for P2
        "aqua": "divelab/aqua"
    }

    model_dict = {
        "Ministral_8B": "mistralai/Ministral-8B-Instruct-2410",
        "Phi_4B": "microsoft/Phi-3-mini-4k-instruct",
        "Llama_8B": "meta-llama/Llama-3.1-8B-Instruct",
        "Gemma_9B": "google/gemma-2-9b-it"

    }
    
    
    
    model_selected = model_dict[model_name]
    
    with open('/home/tmalik6/LLMR/groq_api.txt', 'r', encoding='utf-8') as file:
        groq_key = file.read()
    

    client = Groq(api_key=groq_key)

    dataset_load = dataset_dict.get(dataset_name)
    
    # Load both P1 and P2 datasets
    dataset = input_dataset(dataset_load)
    
    
    
    # Extract questions and answers
    questions_p1 = dataset['train']['question']
    answers_p1 = dataset['train']['answer']
    

    print(f"Aqua dataset size: {len(questions_p1)}")
    
        
    batch_qs = questions_p1
    batch_ans = answers_p1
    model_name = "L70"
    # Run debate-based approach
    df_full = solve(batch_qs, batch_ans, client, model_name=model_name)
    
    # Save results
    df_full.to_csv(f"AQUA_{model_name}_train.csv", index=True)


if __name__ == "__main__":
    main()