import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import re
from transformers import BertTokenizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random

tokenizer_ct = BertTokenizer.from_pretrained("bert-base-uncased")

# =========================
# Config
# =========================
MODEL_DIR = "/home/tmalik6/LLMR/Code/hllm/router_deberta_v3_large_SciNLI/best_model"
MAX_LENGTH = 512
threshold = 0.875

# 0.8675 -> 100% at l8, 0.87 -> 97.2 @ L8, 0.875 -> 0% at l8...


# =========================
# Load model + tokenizer
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model = model.to(device)
model.eval()


@torch.no_grad()
def predict_logit(sent1, sent2, max_length=512):
    enc = tokenizer(
        sent1,
        sent2,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    # Longformer-specific, only use if your router is Longformer
    # global_attention_mask = torch.zeros_like(enc["input_ids"])
    # global_attention_mask[:, 0] = 1
    # enc["global_attention_mask"] = global_attention_mask

    outputs = model(**enc)
    logit = outputs.logits.squeeze(-1).item()
    prob = torch.sigmoid(outputs.logits.squeeze(-1)).item()

    return logit, prob

def clean_cols(df):
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = df.columns.str.strip()
    return df

def input_cost(model_name, input_token_count):
    llama8_inp = 0.05
    llama70_inp = 0.59
    r1_inp = 0.75
    gpt_oss = 0.15
    gpt_oss_small = 0.1
    gpt5_inp = 1.25
    gemma_inp = 0.2
    ministral_inp = 0.1

    if model_name == 'l8':
        cost = llama8_inp
    elif model_name == 'mist':
        cost = ministral_inp
    elif model_name == 'l70':
        cost = llama70_inp
    elif model_name == 'gemini':
        cost = 1.25
    elif model_name == "qwen":
        cost = 0.05
    elif model_name == "phi":
        cost = 0.1
    elif model_name == "phi_r":
        cost = 0.1
    elif model_name == 'gpt_oss_small':
        cost = gpt_oss_small
    elif model_name == "r1":
        cost = r1_inp
    elif model_name == 'gpt_oss':
        cost = gpt_oss
    elif model_name == "gpt5":
        cost = gpt5_inp
    else:
        cost = gemma_inp
        print("input check - Gemma")

    final_cost = cost * input_token_count
    return final_cost


def output_cost(model_name, output_token_count):
    llama8_out = 0.08
    r1_out = 0.99
    llama70_out = 0.79
    gpt_oss_out = 0.6
    gpt_oss_small_out = 0.5
    gpt5_out = 10
    gemma_out = 0.20
    ministral_out = 0.1

    if model_name == 'l8':
        cost = llama8_out
    elif model_name == "phi":
        cost = 0.1
    elif model_name == "qwen":
        cost = 0.4
    elif model_name == "phi_r":
        cost = 0.5
    elif model_name == 'mist':
        cost = ministral_out
    elif model_name == 'l70':
        cost = llama70_out
    elif model_name == "r1":
        cost = r1_out
    elif model_name == "gpt_oss_small":
        cost = gpt_oss_small_out
    elif model_name == 'gpt_oss':
        cost = gpt_oss_out
    elif model_name == "gpt5":
        cost = gpt5_out
    else:
        cost = gemma_out
        print("output check - Gemma")

    final_cost = cost * output_token_count
    return final_cost


def total_cost(model_name, input_length, output_length):
    if model_name == 'gemini':
        return input_cost(model_name, input_length) + (output_length * 10)
    else:
        return input_cost(model_name, input_length) + output_cost(model_name, output_length)


def check_last_letter(s):
    for ch in reversed(str(s).strip()):
        if ch.isalpha():
            ch = ch.lower()
            if ch in ["a", "b", "c", "d"]:
                return ch
            else:
                return None
    return None


def extract_scinli(text):
    lines = [line.strip() for line in str(text).strip().splitlines() if line.strip()]
    if not lines:
        return None

    last_line = lines[-1].lower()

    if "entailment" in last_line:
        return "entailment"
    elif "neutral" in last_line:
        return "neutral"
    elif "contrasting" in last_line:
        return "contrasting"
    elif "reasoning" in last_line:
        return "reasoning"
    else:
        x = check_last_letter(text)
        if x:
            if x == "d":
                return "neutral"
            elif x == "c":
                return "contrasting"
            elif x == "b":
                return "reasoning"
            elif x == "a":
                return "entailment"
        return -1


def count_tokens(string):
    tokens = tokenizer_ct.tokenize(str(string))
    return len(tokens)


def build_scinli_prompt(sent1, sent2):
    user = (
        "Consider the following two sentences:\n"
        f"Sentence1 (premise): {sent1}\n"
        f"Sentence2 (hypothesis): {sent2}\n"
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

    return user


if __name__ == "__main__":

    df_l70 = pd.read_csv("/home/tmalik6/LLMR/Code/scinli/CSVs_latest/SciNLI_L70_new_prompt.csv")
    df_l8 = pd.read_csv("/home/tmalik6/LLMR/Code/scinli/CSVs_latest/SciNLI_L8_new_prompt.csv")
    df = pd.read_csv("/home/tmalik6/LLMR/Code/scinli/test.csv")

    df = clean_cols(df)
    df_l8 = clean_cols(df_l8)
    df_l70 = clean_cols(df_l70)

    # Rename columns so all dfs use same names
    df_l8 = df_l8.rename(columns={
        "Sentence 1": "sentence1",
        "Sentence 2": "sentence2",
        "Correct Answer": "label"
    })

    df_l70 = df_l70.rename(columns={
        "Sentence 1": "sentence1",
        "Sentence 2": "sentence2",
        "Correct Answer": "label"
    })

    # Make sure text columns are strings and stripped
    for curr_df in [df, df_l8, df_l70]:
        curr_df["sentence1"] = curr_df["sentence1"].astype(str).str.strip()
        curr_df["sentence2"] = curr_df["sentence2"].astype(str).str.strip()

    # Create stratified 500-example sample
    sample_df, _ = train_test_split(
        df,
        train_size=500,
        random_state=42,
        stratify=df["label"]
    )

    sample_df = sample_df.reset_index(drop=True)

    # Keep only the columns needed from each model-output dataframe
    df_l8_small = df_l8[["sentence1", "sentence2", "Llama8B"]].copy()
    df_l70_small = df_l70[["sentence1", "sentence2", "Llama70B"]].copy()

    # Merge sample with L8 generations
    big_df = sample_df.merge(
        df_l8_small,
        on=["sentence1", "sentence2"],
        how="left"
    )

    # Merge sample with L70 generations
    big_df = big_df.merge(
        df_l70_small,
        on=["sentence1", "sentence2"],
        how="left"
    )

    # Check if anything failed to match
    print("Missing L8 generations:", big_df["Llama8B"].isna().sum())
    print("Missing L70 generations:", big_df["Llama70B"].isna().sum())

    # Now use these lists in your loop
    l8_full = big_df["Llama8B"].to_list()
    l70_full = big_df["Llama70B"].to_list()

    sent1_full = big_df["sentence1"].to_list()
    sent2_full = big_df["sentence2"].to_list()
    Answers = big_df["label"].to_list()

    print(big_df.shape)
    print(big_df.columns)

    l8_acc = 0
    l70_acc = 0
    hllm_acc = 0

    tok_in = 0
    l8_out = 0
    l70_out = 0

    l8_real = 0
    l8_pred = 0
    l8_pred_cor = 0

    l70_real = 0
    l70_pred = 0
    l70_pred_cor = 0

    hllm_l8 = 0
    l = 0

    r1_acc =0 
    r_l8 = 0

    for l8, l70, sent1, sent2, ans in tqdm(
        zip(l8_full, l70_full, sent1_full, sent2_full, Answers),
        total=len(l8_full)
    ):

        l8 = str(l8)
        l70 = str(l70)

        target = str(ans).lower()
        l8_ans = extract_scinli(l8)
        l70_ans = extract_scinli(l70)

        l += 1

        result = random.randint(0, 1)
        if result == 1:
            rand_ans = l8_ans
            r_l8 +=1
        else:
            rand_ans = l70_ans

        if rand_ans == ans:
            r1_acc +=1
            

        logit, prob = predict_logit(sent1, sent2, max_length=MAX_LENGTH)

        if prob >= threshold:
            hllm_l8 += 1
            hllm_ans = l8_ans
            l8_pred += 1

            if l8_ans == target:
                l8_pred_cor += 1
                hllm_acc += 1

        else:
            hllm_ans = l70_ans
            l70_pred += 1

            if l70_ans == target:
                hllm_acc += 1

            if l8_ans != target and l70_ans == target:
                l70_pred_cor += 1

        if l8_ans == target:
            l8_acc += 1
            l8_real += 1

        if l8_ans != target and l70_ans == target:
            l70_real += 1

        if l70_ans == target:
            l70_acc += 1

        user_baseline = build_scinli_prompt(sent1, sent2)

        tok_in += count_tokens(user_baseline)
        l8_out += count_tokens(l8)
        l70_out += count_tokens(l70)

    print(f"Num of questions: {l}")

    print("\nAccuracy:")
    print(f"l8 acc: {l8_acc/l}")
    print(f"l70 acc: {l70_acc/l}")
    print(f"hllm acc: {hllm_acc/l}")

    tc_l8 = total_cost('l8', tok_in, l8_out) / l
    tc_l70 = total_cost('l70', tok_in, l70_out) / l
    tc_hllm = l8_pred/l * tc_l8 + l70_pred/l * tc_l70

    print("\nCost:")
    print(f"l8 cost: {tc_l8}")
    print(f"l70 cost: {tc_l70}")
    print(f"hllm cost: {tc_hllm}")
    print(f"hllm l8 %: {hllm_l8/l}")

    print("\n Hllm Metrics:")
    print(f"This is threshold: {threshold}")
    # print(f"l8 prec: {l8_pred_cor/l8_pred}")
    # print(f"l8 rec: {l8_pred_cor/l8_real}")
    # print(f"l70 prec: {l70_pred_cor/l70_pred}")
    # print(f"l70 rec: {l70_pred_cor/l70_real}")

    print("\n")
    print("Random 1 Baseline L8 and L70")
    print(f"Random acc {r1_acc/l}")
    r1_tc = tc_l8*r_l8/l + tc_l70*(1-r_l8/l)
    print(f"Random cost: {r1_tc}")