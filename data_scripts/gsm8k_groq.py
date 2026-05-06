from datasets import load_dataset
import pandas as pd
from groq import Groq
import time
import re


def input_dataset(data_name, version=""):
    if version == "":
        dataset = load_dataset(data_name, trust_remote_code=True)
    else:
        dataset = load_dataset(data_name, version, trust_remote_code=True)
    return dataset


def extract_gsm_ans(true_ans):
    """Extract numerical answer from GSM format (#### number)."""
    true_ans = true_ans.replace(",", "")
    match = re.search(r'####\s*([\d.]+)$', true_ans)
    if match:
        number = float(match.group(1)) if "." in match.group(1) else int(match.group(1))
        return number
    return "NA"


def extract_final_answer(model_resp):
    """Extract the last numerical answer from model response."""
    model_resp = model_resp.replace(",", "")
    extracted_num = re.findall(r"-?\d+\.?\d*", model_resp)
    if extracted_num:
        return float(extracted_num[-1])
    return "NA"


def safe_chat_completion(
    client,
    model,
    system_prompt,
    user_prompt,
    temperature=0.0,
    max_tokens=2000,
    retries=3,
    backoff=2,
):
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

            text = response.choices[0].message.content
            return text, None, 1

        except Exception as e:
            err_msg = str(e)
            if "503" in err_msg or "Service Unavailable" in err_msg:
                print(
                    f"[Attempt {attempt + 1}] 503 Service Unavailable — retrying after {backoff} seconds..."
                )
                time.sleep(backoff)
                backoff *= 2
            else:
                return "error", err_msg, 2

    return "error", "failed after 3 errors", 2


def solve_with_debate(qs, ans, client, model_name):
    times_baseline = []

    baseline_full_answers = []
    baseline_extracted_answers = []

    baseline_correct = 0
    correct_answers = []
    failed_target_extract = 0

    system = "You are a helpful assistant."

    for i in range(len(qs)):
        print(f"Question {i + 1} in progress...")

        target_ans = extract_gsm_ans(ans[i])

        if target_ans == "NA":
            target_ans = "NA_target"
            print("****")
            print(f"This is the failed extract ans (i+1): {i+1}")
            print(f"This is full text ans: {ans[i]}")
            print("****")
            failed_target_extract += 1

        correct_answers.append(target_ans)

        user_baseline = (
            "As an expert problem solver, solve step by step the following mathematical question.\n\n"
        )
        user_baseline += f"Q: {qs[i]}\nA: Let's think step by step.\n\n"
        user_baseline += (
            "Use the following template for structuring your answer:\n\n"
            "Step 1:\n\n"
            "Step 2:\n\n"
            "...\n\n"
            "Last Step:\n\n"
            "Final numerical answer: ..."
        )

        time_start = time.time()
        text_baseline, err, hit = safe_chat_completion(
            client,
            model_name,
            system,
            user_baseline,
            temperature=0.5,
            max_tokens=4096,
        )
        time_end = time.time()
        times_baseline.append(time_end - time_start)

        if hit == 2:
            print(f"Groq error: {err}")
            text_baseline = "NA_llama_70b"

        baseline_full_answers.append(text_baseline)

        extracted_baseline = extract_final_answer(text_baseline)
        if extracted_baseline == "NA":
            extracted_baseline = "NA_llama_70b"

        baseline_extracted_answers.append(extracted_baseline)

        is_baseline_correct = extracted_baseline == target_ans
        if is_baseline_correct:
            baseline_correct += 1

        print(f"Question {i + 1} completed. Baseline: {is_baseline_correct}")

    df_results = pd.DataFrame(
        {
            "Question": qs,
            "Correct Answer": correct_answers,
            "Baseline Answer": baseline_extracted_answers,
            "Baseline Correct": [
                baseline_extracted_answers[i] == correct_answers[i]
                for i in range(len(correct_answers))
            ],
        }
    )

    df_full = pd.DataFrame(
        {
            "Question": qs,
            "Correct Answer": ans,
            "Baseline Full": baseline_full_answers,
        }
    )

    total_valid = len(correct_answers)
    print("\n===== GSM8K EXPERIMENT RESULTS =====")
    print(f"Total valid questions: {total_valid}")
    print(f"Failed target extractions: {failed_target_extract}")

    print("\n----- Accuracy -----")
    print(f"Baseline accuracy: {baseline_correct / total_valid:.4f} ({baseline_correct}/{total_valid})")

    print("\n----- Average Response Times -----")
    print(f"Baseline ({model_name}): {sum(times_baseline) / len(times_baseline):.2f}s")

    return df_results, df_full


def main():
    dataset_name = "gsm8k"

    dataset_dict = {
        "gsm8k": "openai/gsm8k",
        "GSM": "apple/GSM-Symbolic",
    }

    model_name = "llama-3.3-70b-versatile"

    with open("/home/tmalik6/LLMR/groq_api.txt") as file:
        api_k = file.read().strip()

    client = Groq(api_key=api_k)

    dataset_load = dataset_dict.get(dataset_name)
    dataset_p1 = input_dataset(dataset_load, "main")

    # USE TRAIN SET INSTEAD OF TEST SET
    questions_p1 = dataset_p1["train"]["question"]
    answers_p1 = dataset_p1["train"]["answer"]

    print(f"GSM8K training set size: {len(questions_p1)}")

    batch_qs = questions_p1
    batch_ans = answers_p1

    df_results, df_full = solve_with_debate(batch_qs, batch_ans, client, model_name)

    df_results.to_csv("GSM8K_training_llama_70b_results.csv", index=False)
    df_full.to_csv("GSM8K_training_llama_70b_full.csv", index=False)


if __name__ == "__main__":
    main()