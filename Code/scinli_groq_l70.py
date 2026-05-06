import pandas as pd
from groq import Groq
import time
from datasets import  load_dataset 

def input_dataset(data_name, version=""):
    if version == "":
        dataset = load_dataset(data_name)
    else:
        dataset = load_dataset(data_name, version)
    return dataset

def groq_one_sample(client, user, system, temperature=0.7, max_tokens=2048,
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


def solve(sent1, sent2, label, client, sc = 1):
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
        full_sc, x = groq_one_sample(client, user, system, temperature=0.7, max_tokens=2048,retries=3, backoff=2)
        time_end = time.time()
        
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
        df_data[f"Llama70B SC ({i+1})"] = full_l8[i]
    
    df_debate_full = pd.DataFrame(df_data)
    
    
    print("\n----- Average Response Times -----")
    print(f"{sum(times)/len(times):.2f}s")
    print()
    
    return df_debate_full
def main():


    with open("/home/tmalik6/LLMR/groq_api.txt") as file:
        api_k = file.read()

    client = Groq(api_key=api_k)

    dataset = input_dataset("tasksource/scinli")

    
    sent1 = dataset['train']["sentence1"][:8000]
    sent2 = dataset['train']["sentence2"][:8000]
    labels = dataset['train']["label"][:8000]

    df = solve(sent1, sent2, labels, client, 1)
    df.to_csv('SciNLI_L70_train_8k.csv', index=True)

    

main()