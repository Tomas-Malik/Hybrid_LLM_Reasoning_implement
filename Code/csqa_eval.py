import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import re
from transformers import BertTokenizer
from tqdm import tqdm
from datasets import load_dataset

tokenizer_ct = BertTokenizer.from_pretrained("bert-base-uncased")

# =========================
# Config
# =========================
MODEL_DIR = "/home/tmalik6/LLMR/Code/hllm/router_deberta_v3_large_CSQA/best_model"
MAX_LENGTH = 512
threshold = 0.875

# =========================
# Load model + tokenizer
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model = model.to(device)
model.eval()


@torch.no_grad()
def predict_logit_bert(text, max_length=256):
    """
    Returns:
        logit: raw model output (float)
        prob: sigmoid(logit), optional convenience score
    """
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    outputs = model(**enc)
    logit = outputs.logits.squeeze(-1).item()
    prob = torch.sigmoid(outputs.logits.squeeze(-1)).item()

    return logit, prob


@torch.no_grad()
def predict_logit(text, max_length=512):
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    # Longformer-specific
    # global_attention_mask = torch.zeros_like(enc["input_ids"])
    # global_attention_mask[:, 0] = 1
    # enc["global_attention_mask"] = global_attention_mask

    outputs = model(**enc)
    logit = outputs.logits.squeeze(-1).item()
    prob = torch.sigmoid(outputs.logits.squeeze(-1)).item()

    return logit, prob


@torch.no_grad()
def predict_logits_bert(texts, max_length=256):
    """
    Batch version for multiple texts.

    Returns:
        results: list of dicts with text, logit, prob
    """
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    outputs = model(**enc)
    logits = outputs.logits.squeeze(-1)
    probs = torch.sigmoid(logits)

    results = []
    for text, logit, prob in zip(texts, logits.tolist(), probs.tolist()):
        results.append({
            "text": text,
            "logit": logit,
            "prob": prob,
        })

    return results


@torch.no_grad()
def predict_logits(texts, max_length=256):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    # Longformer-specific
    # global_attention_mask = torch.zeros_like(enc["input_ids"])
    # global_attention_mask[:, 0] = 1
    # enc["global_attention_mask"] = global_attention_mask

    outputs = model(**enc)
    logits = outputs.logits.squeeze(-1)
    probs = torch.sigmoid(logits)

    results = []
    for text, logit, prob in zip(texts, logits.tolist(), probs.tolist()):
        results.append({
            "text": text,
            "logit": logit,
            "prob": prob,
        })

    return results


def input_dataset(data_name, version=""):
    if version == "":
        dataset = load_dataset(data_name)
    else:
        dataset = load_dataset(data_name, version)
    return dataset


def input_cost(model_name, input_token_count):
    llama8_inp = 0.05
    llama70_inp = 0.59
    r1_inp = 0.75 #R1
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
    elif model_name == "phi":
        cost = 0.1
    elif model_name == "phi_r":
        cost = 0.1
    elif model_name == "qwen":
        cost = 0.05
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
    llama8_inp = 0.08
    r1_inp = 0.99 #R1
    llama70_inp = 0.79 
    gpt_oss = 0.6
    gpt_oss_small = 0.5
    gpt5_inp = 10
    gemma_inp = 0.20
    ministral_inp = 0.1
    if model_name == 'l8':
        cost = llama8_inp
    elif model_name == "phi":
        cost = 0.1
    elif model_name == "phi_r":
        cost = 0.5
    elif model_name == "qwen":
        cost = 0.4
    elif model_name == 'mist':
        cost = ministral_inp
    elif model_name == 'l70':
        cost = llama70_inp
    elif model_name == "r1":
        cost = r1_inp
    elif model_name == "gpt_oss_small":
        cost = gpt_oss_small
    elif model_name == 'gpt_oss':
        cost = gpt_oss
    elif model_name == "gpt5":
        cost = gpt5_inp
    else:
        cost = gemma_inp
        print("output check - Gemma")
    
    final_cost = cost * output_token_count
    return final_cost


def total_cost(model_name, input_length, output_length):
    if model_name == 'gemini':
        return (input_cost(model_name, input_length) + (output_length * 10))
    else:
        return (input_cost(model_name, input_length) + output_cost(model_name, output_length))


def extract_ans(text):
    lines = str(text).strip().splitlines()
    
    # Iterate backwards to find the first non-empty line
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    
    return None


def preprocess_options(options):
    otp = []
    for i, j in zip(options["label"], options["text"]):
        x = f"{i}. {j}"
        otp.append(x)
    return otp


def ans_proc(full_ans, options):
    flag = False
    letters = options["label"]
    words = options["text"]
    last_line = extract_ans(full_ans)

    if last_line is None:
        return "NA"

    for x, j in enumerate(words):
        if j.lower() in last_line.lower():
            ans = letters[x]
            flag = True
            break

    if flag == False:
        for j in reversed(letters):
            if j in last_line:
                ans = j
                flag = True
                break

    if flag == False:
        ans = "NA"

    return ans


def count_tokens(string):
    tokens = tokenizer_ct.tokenize(str(string))
    return len(tokens)


if __name__ == "__main__":

    df = pd.read_csv("/home/tmalik6/LLMR/Code/CommonSenseQA/CSVs_latest/CSQA_FULL.csv")
    # df1 = pd.read_csv("/home/tmalik6/LLMR/Code/CommonSenseQA/CSVs_latest/CSQA_SC_local_l8_n7.csv")

    dataset = input_dataset("tau/commonsense_qa")
    opts = dataset["validation"]["choices"]
    Questions = dataset["validation"]["question"]
    Answers = dataset["validation"]["answerKey"]

    l8_full = df["Llama8b"].to_list()
    l70_full = df["Llama70b_greedy"].to_list()

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

    for l8, l70, q, ans, opt_curr in tqdm(
        zip(l8_full, l70_full, Questions, Answers, opts),
        total=len(l8_full)
    ):

        l8 = str(l8)
        l70 = str(l70)

        target = str(ans)
        l8_ans = ans_proc(l8, opt_curr)
        l70_ans = ans_proc(l70, opt_curr)

        l += 1

        logit, prob = predict_logit(q, max_length=MAX_LENGTH)

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

        user_baseline = "As an expert problem solver, solve step by step the following question and choose the right answer from the presented options.\n\n"
        user_baseline += f"Q: {q}\nA: Let's think step by step.\n\n"
        user_baseline += f"The options are: {opt_curr}. Indicate the the latter and the word of the option you're choosing as the final answer in the following format: Final Answer: 'Letter'. 'Word'"
        
        tok_in += count_tokens(user_baseline)
        l8_out += count_tokens(l8)
        l70_out += count_tokens(l70)

    
    print(f"Num of questions: {l}")

    print("\nAccuracy:")
    print(f"l8 acc: {l8_acc/l}")
    print(f"l70 acc: {l70_acc/l}")
    print(f"hllm acc: {hllm_acc/l}")

    tc_l8 = total_cost('l8', tok_in, l8_out)/l
    tc_l70 = total_cost('l70', tok_in, l70_out)/l
    tc_hllm = l8_pred/l*tc_l8 + l70_pred/l*tc_l70

    print("\nCost:")
    print(f"l8 cost: {tc_l8}")
    print(f"l70 cost: {tc_l70}")
    print(f"hllm cost: {tc_hllm}")
    print(f"hllm l8 %: {hllm_l8/l}")

    print("\n Hllm Metrics:")
    print(f"This is threshold: {threshold}")
    print(f"l8 prec: {l8_pred_cor/l8_pred}")
    print(f"l8 rec: {l8_pred_cor/l8_real}")
    print(f"l70 prec: {l70_pred_cor/l70_pred}")
    print(f"l70 rec: {l70_pred_cor/l70_real}")