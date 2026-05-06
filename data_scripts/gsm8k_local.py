from datasets import load_dataset 
import pandas as pd
import time
import re
from vllm import LLM
from vllm.sampling_params import SamplingParams

def input_dataset(data_name, version=""):
    if version == "":
        dataset = load_dataset(data_name, trust_remote_code=True)
    else:
        dataset = load_dataset(data_name, version, trust_remote_code=True)
    return dataset

def phi_engine(system_prompt, user_prompt,llm):
    
    request = f"<|system|>{system_prompt}<|end|><|user|>{user_prompt}<|end|><|assistant|>" # https://huggingface.co/microsoft/Phi-4-mini-instruct
    sampling_params = SamplingParams(temperature=0.0,seed=42,max_tokens=2000) 
    outputs = llm.generate(request, sampling_params,use_tqdm=False)
    return outputs

def extract_gsm_ans(true_ans):
    """Extract numerical answer from GSM format (#### number)"""
    true_ans = true_ans.replace(",", "")
    match = re.search(r'####\s*([\d.]+)$', true_ans)
    if match:
        number = float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
        return number
    return "NA"

def extract_final_answer(model_resp):
    """Extract the last numerical answer from model response"""
    # Remove commas so for example 5,000 becomes 5000
    model_resp = model_resp.replace(",", "")
    # Find the last number
    extracted_num = re.findall(r"-?\d+\.?\d*", model_resp)
    if extracted_num:
        return float(extracted_num[-1])
    else:
        return "NA"

def safe_chat_completion(client, model, system_prompt, user_prompt, temperature=0.0, max_tokens=2000, retries=3, backoff=2):
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
    return "error", "failed after 3 errors"

def solve_with_debate(qs, ans, client):
    # For tracking performance
    times_baseline = []
    times_debate_round1 = []
    times_debate_round2 = []
    
    # For storing answers
    baseline_full_answers = []
    baseline_extracted_answers = []
    
    
    # For tracking accuracy
    baseline_correct = 0
    
    correct_answers = []  # Extracted target answers
    
    # Skip tracking
    skipped_qs = []
    failed_target_extract = 0
    
    system = "You are a helpful assistant."
    samp_params = SamplingParams(temperature=0.5,seed=42,max_tokens=2048)
    # Process each question
    for i in range(len(qs)):
        print(f'Question {i + 1} in progress...')
        
        # Extract target answer
        target_ans = extract_gsm_ans(ans[i])
        
        if target_ans == "NA":
            target_ans = "NA_target"
            print("****")
            print(f"This is the failed extract ans (i+1):{i+1}")
            print(f"this is full text ans: {ans[i]}")
            print("Question skipped")
            print("****")
            failed_target_extract += 1
            
        
        correct_answers.append(target_ans)
        
        # BASELINE: 
        user_baseline = "As an expert problem solver, solve step by step the following mathematical question.\n\n"
        user_baseline += f'Q: {qs[i]}\nA: Let\'s think step by step.\n\n'
        user_baseline += "Use the following template for structuring your answer: \n\n Step 1: \n\n Step 2: \n\n ... \n\n Last Step: \n\n Final numerical answer: ..."
        
        messages = [
            {
                "role": "system",
                "content": system
            },
            {
                "role": "user",
                "content": user_baseline
            },
        ]

        time_start = time.time()
        outputs = client.chat(messages, sampling_params=samp_params)
        text_baseline = outputs[0].outputs[0].text
        time_end = time.time()
        times_baseline.append(time_end - time_start)
        
        baseline_full_answers.append(text_baseline)
        extracted_baseline = extract_final_answer(text_baseline)
        if extracted_baseline == "NA":
            extracted_baseline = "NA_mistral"
        baseline_extracted_answers.append(extracted_baseline)
        
        # Check if baseline is correct
        is_baseline_correct = (extracted_baseline == target_ans)
        if is_baseline_correct:
            baseline_correct += 1
        
        
        print(f"Question {i+1} completed. Baseline: {is_baseline_correct}")
    
    # Create DataFrame with results
    df_debate_results = pd.DataFrame({
        "Question": qs,
        "Correct Answer": correct_answers,
        "Baseline Answer": baseline_extracted_answers,
        "Baseline Correct": [baseline_extracted_answers[i] == correct_answers[i] for i in range(len(correct_answers))],
        })
    
    # Create DataFrame with full answers
    df_debate_full = pd.DataFrame({
        "Question": qs,
        "Correct Answer": ans,
        "Baseline Full": baseline_full_answers,
    })
    
    # Print summary statistics
    total_valid = len(correct_answers)
    print("\n===== GSM-SYMBOLIC DEBATE EXPERIMENT RESULTS =====")
    print(f"Total valid questions: {total_valid}")
    print(f"Failed target extractions: {failed_target_extract}")
    
    print("\n----- Accuracy -----")
    print(f"Baseline accuracy: {baseline_correct/total_valid:.4f} ({baseline_correct}/{total_valid})")
    
   
    print("\n----- Average Response Times -----")
    print(f"Baseline (Gemma2-9B): {sum(times_baseline)/len(times_baseline):.2f}s")
    print(f"Total debate time per question: {(sum(times_baseline) + sum(times_debate_round1) + sum(times_debate_round2))/len(times_baseline):.2f}s")
    
    return df_debate_results, df_debate_full

def main():
    dataset_name = "gsm8k"
    
    dataset_dict = {
        "gsm8k": "openai/gsm8k", # 7.47k training size, 1.32k test size
        "GSM": "apple/GSM-Symbolic" # 5k test set for P1, 2.5k for P2
    }
    
    # Read API key from file or set directly
    # with open("groq_api.txt") as file:
    #     api_k = file.read()
    # api_k = "your_api_key_here"  # Alternative: set directly
    
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    client = LLM(model=model_name,gpu_memory_utilization=0.9,max_model_len=2000,tensor_parallel_size=1) 

    
    
    dataset_load = dataset_dict.get(dataset_name)
    
    
    dataset_p1 = input_dataset(dataset_load, 'main')
    
    
    # Extract questions and answers
    questions_p1 = dataset_p1['train']['question']
    answers_p1 = dataset_p1['train']['answer']
    
    
    
    print(f"GSM8K dataset size: {len(questions_p1)}")
    
        
    batch_qs = questions_p1
    batch_ans = answers_p1
    
    # Run debate-based approach
    df_full = solve_with_debate(batch_qs, batch_ans, client)
    
    df_full.to_csv(f"AQUA_training_l8.csv", index=True)
    
    

if __name__ == "__main__":
    main()