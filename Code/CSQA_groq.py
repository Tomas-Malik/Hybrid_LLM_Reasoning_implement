from datasets import  load_dataset 
import pandas as pd
from groq import Groq
import time

def input_dataset(data_name, version = ""):
    if version == "":
        dataset = load_dataset(data_name, trust_remote_code=True)
    else:
        dataset = load_dataset(data_name,version, trust_remote_code=True)
    return dataset

def extract_ans(text):
    lines = text.strip().splitlines()
    
    # Iterate backwards to find the first non-empty line
    for line in reversed(lines):
        if line.strip():  # Non-empty after stripping spaces
            return line.strip()
    
    return None  # No non-empty lines found
        
    
def safe_chat_completion(client, model, system_prompt, user_prompt, temperature=0.5, max_tokens=2000, retries=3, backoff=2):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=model,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            return response.choices[0].message.content, None
        except Exception as e:
            if "503" in str(e) or "Service Unavailable" in str(e):
                print(f"[Attempt {attempt+1}] 503 Service Unavailable — retrying after {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2  # exponential backoff
            else:
                return "error", str(e)
    return "error","failed after 3 errors"

def preprocess_options(options):
    otp = []
    for i, j in zip(options["label"], options["text"]):
        x = f"{i}. {j}"
        otp.append(x)
    return otp

def solve(qs, options, ans, client):
    
    times_l70bg = []
    times_l8b = []
    
    full_l70bg = []
    full_l8b = []

    l = len(qs)
    system = "You are a helpful assistant."
    for i in range(l):
        
        print(f'Question {i + 1} in progress...')
        
        options_curr = preprocess_options(options[i])
        
        user = "As an expert problem solver, solve step by step the following question and choose the right answer from the presented options.\n\n"
        user = user + f'Q: {qs[i]}\nA: Let\'s think step by step.\n\n'
        user = user + f"The options are: {options_curr}. Indicate the the latter and the word of the option you're choosing as the final answer in the following format: Final Answer: 'Letter'. 'Word'"
    
        

        # llama 3.3g
        time_start = time.time()
        text_llama70b, error = safe_chat_completion(client, model="llama-3.3-70b-versatile", system_prompt=system,user_prompt= user)
        time_end = time.time()
        times_l70bg.append(time_end - time_start)
        full_l70bg.append(text_llama70b)
        

    df_full = pd.DataFrame({
        "Question": qs,
        "Answer": ans,
        "L70": full_l70bg
    })

    # print("***** Average times *****")
    # avg_time_l70b_g = sum(times_l70bg) / (len(times_l70bg))
    # avg_time_l8b = sum(times_l8b) / (len(times_l8b))
    # print(f"Average query time for Qwen: {avg_time_l70b_g} (Groq)")
    # print(f"Average query time for Gemma2: {avg_time_l8b} (Groq)")
    

    # print times should be added

    return df_full

def main():

    dataset_name="CSQA"



    dataset_dict = {
        "gsm8k": "openai/gsm8k", # 7.47k training size, 1.32k test size
        "MATHold": "lighteval/MATH", #outdated?
        "MATH": "xDAN2099/lighteval-MATH",
        "MATH500":  "HuggingFaceH4/MATH-500",
         "CSQA": "tau/commonsense_qa"
    }

    with open("/home/tmalik6/LLMR/groq_api.txt") as file:
        api_k = file.read()

    client = Groq(api_key=api_k)


    dataset_load = dataset_dict.get(dataset_name)
    dataset=input_dataset(dataset_load)


    questions = dataset['train']['question']
    options = dataset['train']['choices']
    answers = dataset['train']['answerKey']

    df2 = solve(questions, options, answers, client)
    df2.to_csv('CSQA_train_L70.csv', index=True)

    

main()