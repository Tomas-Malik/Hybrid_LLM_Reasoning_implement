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


def extract_math_ans(true_ans):
    init = len(true_ans) - 1
    targ_len = len("\\boxed")
    final_ans = ""

    while (init - targ_len) >= 0:
        if true_ans[init - targ_len:init] == "\\boxed":
            break
        init -= 1
    if init - targ_len < 0:
        return "NA"

    start = init
    while start < len(true_ans) and true_ans[start] != "{":
        start += 1
    if start == len(true_ans):
        return "NA"

    start += 1
    brackets = 1
    for i in range(start, len(true_ans)):
        if true_ans[i] == "{":
            brackets += 1
        elif true_ans[i] == "}":
            brackets -= 1
            if brackets == 0:
                return final_ans
        final_ans += true_ans[i]

    return "NA"


def safe_chat_completion(
    client,
    model,
    system_prompt,
    user_prompt,
    temperature=0.5,
    max_tokens=2048,
    retries=3,
    backoff=2,
):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            return response.choices[0].message.content, None, 1

        except Exception as e:
            err_msg = str(e)
            if "503" in err_msg or "Service Unavailable" in err_msg:
                print(f"[Attempt {attempt + 1}] 503 error — retrying after {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2
            else:
                return "error", err_msg, 2

    return "error", "failed after 3 retries", 2


def self_consistency_chat(
    client,
    model,
    system_prompt,
    user_prompt,
    n,
    temperature=0.5,
    max_tokens=2048,
):
    completions = []

    for _ in range(n):
        text, err, hit = safe_chat_completion(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if hit == 2:
            print(f"Groq error: {err}")
            text = "NA_L70groq"

        completions.append(text)

    return completions


def solve(qs, ans, client, model_name, sc=1):
    times = []
    full_l8 = [[] for _ in range(sc)]

    system = (
        "You are a helpful assistant that interacts entirely in LaTeX code. "
        "All responses should be formatted in LaTeX, including explanations, equations, and text."
    )

    for i in range(len(qs)):
        print(f"Question {i + 1} in progress...")

        user = "As an expert problem solver, solve step by step the following mathematical question.\n\n"
        user += f"Q: {qs[i]}\nA: Let's think step by step.\n\n"
        user += (
            "Use the following template for structing your answer:\n\n"
            "Step 1:\n\n"
            "Step 2:\n\n"
            "...\n\n"
            "Last Step:\n\n"
            "Final answer: \\boxed{...}"
        )

        time_start = time.time()
        full_sc = self_consistency_chat(
            client=client,
            model=model_name,
            system_prompt=system,
            user_prompt=user,
            n=sc,
            temperature=0.5,
            max_tokens=4096,
        )
        time_end = time.time()

        for idx, sample in enumerate(full_sc):
            full_l8[idx].append(sample)

        times.append(time_end - time_start)

    df_data = {
        "Question": qs,
        "Correct Answer": ans,
    }

    for i in range(sc):
        df_data[f"Llama3.1-8B SC ({i+1})"] = full_l8[i]

    df_debate_full = pd.DataFrame(df_data)

    print("\n----- Average Response Times -----")
    print(f"{sum(times) / len(times):.2f}s")
    print()

    return df_debate_full


def main():
    dataset_name = "MATH"

    dataset_dict = {
        "gsm8k": "openai/gsm8k",
        "MATHold": "lighteval/MATH",
        "MATH": "xDAN2099/lighteval-MATH",
        "MATH500": "HuggingFaceH4/MATH-500",
    }

    dataset_load = dataset_dict.get(dataset_name)
    dataset = input_dataset(dataset_load)

    # train split, same as your current local script
    questions = dataset["train"]["problem"]
    answers = dataset["train"]["solution"]

    # Read Groq API key
    with open("/home/tmalik6/LLMR/groq_api.txt", "r") as f:
        api_key = f.read().strip()

    client = Groq(api_key=api_key)

    # Same model family/size as your local setup, but Groq-hosted
    model_name = "llama-3.3-70b-versatile"

    df = solve(questions, answers, client, model_name, sc=1)
    df.to_csv("MATH_train_llama_70B_groq.csv", index=True)


if __name__ == "__main__":
    main()