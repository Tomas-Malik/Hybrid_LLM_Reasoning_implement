import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import re
from typing import Optional, Dict
from transformers import BertTokenizer
from tqdm import tqdm


# =========================
# Config
# =========================

# Change this to your AQuA router model path
MODEL_DIR = "/home/tmalik6/LLMR/Code/hllm/router_deberta_v3_large_AQuA/best_model"

MAX_LENGTH = 512
threshold = 0.83

# AQuA CSV paths
L8_CSV_PATH = "/home/tmalik6/LLMR/Code/aqua/CSVs_latest/AQUA_Llama_8B_full_v2.csv"
L70_CSV_PATH = "/home/tmalik6/LLMR/Code/aqua/CSVs_latest/AQUA_Llama_70B_full.csv"


# =========================
# Tokenizers
# =========================

tokenizer_ct = BertTokenizer.from_pretrained("bert-base-uncased")


# =========================
# Load router model + tokenizer
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model = model.to(device)
model.eval()


# =========================
# Router prediction functions
# =========================

@torch.no_grad()
def predict_logit_bert(text, max_length=512):
    """
    Standard encoder classifier version.
    Use this for DeBERTa/BERT/RoBERTa-style routers.

    Returns:
        logit: raw model output
        prob: sigmoid(logit)
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
def predict_logit_longformer(text, max_length=512):
    """
    Longformer-specific version.
    Use this only if your router is Longformer.
    """

    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    global_attention_mask = torch.zeros_like(enc["input_ids"])
    global_attention_mask[:, 0] = 1
    enc["global_attention_mask"] = global_attention_mask

    outputs = model(**enc)
    logit = outputs.logits.squeeze(-1).item()
    prob = torch.sigmoid(outputs.logits.squeeze(-1)).item()

    return logit, prob


# Choose router function here
predict_logit = predict_logit_bert
# predict_logit = predict_logit_longformer


# =========================
# Cost functions
# =========================

def input_cost(model_name, input_token_count):
    llama8_inp = 0.05
    llama70_inp = 0.59
    r1_inp = 0.75
    gpt_oss = 0.15
    gpt_oss_small = 0.1
    gpt5_inp = 1.25
    gemma_inp = 0.2
    ministral_inp = 0.1

    if model_name == "l8":
        cost = llama8_inp
    elif model_name == "mist":
        cost = ministral_inp
    elif model_name == "l70":
        cost = llama70_inp
    elif model_name == "gemini":
        cost = 1.25
    elif model_name == "qwen":
        cost = 0.05
    elif model_name == "phi":
        cost = 0.1
    elif model_name == "phi_r":
        cost = 0.1
    elif model_name == "gpt_oss_small":
        cost = gpt_oss_small
    elif model_name == "r1":
        cost = r1_inp
    elif model_name == "gpt_oss":
        cost = gpt_oss
    elif model_name == "gpt5":
        cost = gpt5_inp
    else:
        cost = gemma_inp
        print("input check - Gemma")

    return cost * input_token_count


def output_cost(model_name, output_token_count):
    llama8_out = 0.08
    r1_out = 0.99
    llama70_out = 0.79
    gpt_oss_out = 0.6
    gpt_oss_small_out = 0.5
    gpt5_out = 10
    gemma_out = 0.20
    ministral_out = 0.1

    if model_name == "l8":
        cost = llama8_out
    elif model_name == "phi":
        cost = 0.1
    elif model_name == "qwen":
        cost = 0.4
    elif model_name == "phi_r":
        cost = 0.5
    elif model_name == "mist":
        cost = ministral_out
    elif model_name == "l70":
        cost = llama70_out
    elif model_name == "r1":
        cost = r1_out
    elif model_name == "gpt_oss_small":
        cost = gpt_oss_small_out
    elif model_name == "gpt_oss":
        cost = gpt_oss_out
    elif model_name == "gpt5":
        cost = gpt5_out
    else:
        cost = gemma_out
        print("output check - Gemma")

    return cost * output_token_count


def total_cost(model_name, input_length, output_length):
    if model_name == "gemini":
        return input_cost(model_name, input_length) + (output_length * 10)
    else:
        return input_cost(model_name, input_length) + output_cost(model_name, output_length)


# =========================
# AQuA extraction functions
# =========================

def parse_aqua_options(question: str) -> Dict[str, str]:
    matches = re.findall(r"([A-E])\)\s*([^\s]+)", str(question))
    return {letter: value for letter, value in matches}


def extract_last_number(text: str) -> Optional[str]:
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(text))
    return nums[-1] if nums else None


def extract_final_letter(model_output: str, question: str) -> Optional[str]:
    """
    Returns:
        Single character string: A/B/C/D/E
        or "fail"
    """

    model_output = str(model_output)

    # 1. Full format: Final Answer: 'C'. '42'
    match = re.search(
        r"Final\s*Answer\s*:\s*['\"]?([A-E])['\"]?\s*\.\s*['\"]?-?\d+(?:\.\d+)?['\"]?",
        model_output,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    # 2. Letter-only format: Final Answer: C
    match = re.search(
        r"Final\s*Answer\s*:\s*['\"]?([A-E])['\"]?",
        model_output,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    # 3. Backup: last number -> map to options
    last_num = extract_last_number(model_output)
    if last_num:
        options = parse_aqua_options(question)
        for letter, value in options.items():
            if value == last_num:
                return letter

    # print("failed to extract")
    return "fail"


def normalize_target_answer(ans):
    """
    AQuA target is usually already a letter A/B/C/D/E.
    """

    ans = str(ans).strip().upper()

    if ans in ["A", "B", "C", "D", "E"]:
        return ans

    match = re.search(r"\b([A-E])\b", ans)
    if match:
        return match.group(1)

    return "NA_target"


# =========================
# Token counting
# =========================

def count_tokens(string):
    tokens = tokenizer_ct.tokenize(str(string))
    return len(tokens)


def build_aqua_prompt(q):
    """
    Matches the prompt style used in your AQuA analysis script.
    """

    user_baseline = "You are a helpful assistant."
    user_baseline += "As an expert problem solver, solve step by step the following mathematical question.\n\n"
    user_baseline += f"Q: {q}\nA: Let's think step by step.\n\n"
    user_baseline += (
        "Indicate the the latter and the word of the option you're choosing as "
        "the final answer in the following format: Final Answer: 'Letter'. 'number'"
    )

    return user_baseline


def safe_div(num, den):
    if den == 0:
        return 0.0
    return num / den


# =========================
# Main
# =========================

if __name__ == "__main__":

    # =========================
    # Load AQuA outputs
    # =========================

    df_l8 = pd.read_csv(L8_CSV_PATH)
    df_l70 = pd.read_csv(L70_CSV_PATH)

    df_l8 = df_l8.loc[:, ~df_l8.columns.str.contains("^Unnamed")]
    df_l70 = df_l70.loc[:, ~df_l70.columns.str.contains("^Unnamed")]

    df_l8.columns = df_l8.columns.str.strip()
    df_l70.columns = df_l70.columns.str.strip()

    # print("L8 columns:")
    # print(df_l8.columns.tolist())

    # print("\nL70 columns:")
    # print(df_l70.columns.tolist())

    # Expected columns from your AQuA script
    questions = df_l8["Question"].to_list()
    targets = df_l8["Correct Answer"].to_list()

    l8_full = df_l8["Llama_8B Full"].to_list()
    l70_full = df_l70["Llama_70B Full"].to_list()

    # =========================
    # Metrics
    # =========================

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

    fail_l8 = 0
    fail_l70 = 0
    fail_target = 0

    l = 0

    routed_rows = []

    # =========================
    # Eval loop
    # =========================

    for l8, l70, q, target in tqdm(
        zip(l8_full, l70_full, questions, targets),
        total=len(questions)
    ):
        l += 1

        l8 = str(l8)
        l70 = str(l70)
        target_ans = normalize_target_answer(target)

        l8_ans = extract_final_letter(l8, q)
        l70_ans = extract_final_letter(l70, q)

        if target_ans == "NA_target":
            fail_target += 1

        if l8_ans == "fail":
            l8_ans = "NA_l8"
            fail_l8 += 1

        if l70_ans == "fail":
            l70_ans = "NA_l70"
            fail_l70 += 1

        # Router score
        logit, prob = predict_logit(q, max_length=MAX_LENGTH)

        # HybridLLM decision:
        # prob >= threshold means route to small model L8
        if prob >= threshold:
            hllm_l8 += 1
            hllm_ans = l8_ans
            route = "l8"
            l8_pred += 1

            if l8_ans == target_ans:
                l8_pred_cor += 1
                hllm_acc += 1

        else:
            hllm_ans = l70_ans
            route = "l70"
            l70_pred += 1

            if l70_ans == target_ans:
                hllm_acc += 1

            # This is the positive class for routing to L70:
            # L8 is wrong and L70 is correct.
            if l8_ans != target_ans and l70_ans == target_ans:
                l70_pred_cor += 1

        # Individual model accuracy
        if l8_ans == target_ans:
            l8_acc += 1
            l8_real += 1

        if l70_ans == target_ans:
            l70_acc += 1

        # Cases where L70 is useful over L8
        if l8_ans != target_ans and l70_ans == target_ans:
            l70_real += 1

        # Token counting
        user_baseline = build_aqua_prompt(q)
        tok_in += count_tokens(user_baseline)

        l8_out += count_tokens(l8)
        l70_out += count_tokens(l70)

        routed_rows.append({
            "Question": q,
            "Correct Answer": target_ans,
            "L8 Extracted": l8_ans,
            "L70 Extracted": l70_ans,
            "Router Logit": logit,
            "Router Prob": prob,
            "Route": route,
            "Hybrid Answer": hllm_ans,
            "Hybrid Correct": int(hllm_ans == target_ans),
            "L8 Correct": int(l8_ans == target_ans),
            "L70 Correct": int(l70_ans == target_ans),
        })

    # =========================
    # Cost
    # =========================

    tc_l8 = total_cost("l8", tok_in, l8_out) / l
    tc_l70 = total_cost("l70", tok_in, l70_out) / l

    # Same cost accounting style as your Math500 HybridLLM script:
    # weighted average of the selected model costs.
    tc_hllm = (l8_pred / l) * tc_l8 + (l70_pred / l) * tc_l70

    # =========================
    # Save routed dataframe
    # =========================

    df_routes = pd.DataFrame(routed_rows)
    # out_path = f"AQUA_HybridLLM_threshold_{threshold}.csv"
    # df_routes.to_csv(out_path, index=False)

    # =========================
    # Print results
    # =========================

    print(f"\nNum of questions: {l}")

    print("\nAccuracy:")
    print(f"l8 acc: {l8_acc / l}")
    print(f"l70 acc: {l70_acc / l}")
    print(f"hllm acc: {hllm_acc / l}")

    print(f"\nSANITY CHECK: {(l8_pred + l70_pred) == l}")

    print("\nCost:")
    print(f"l8 cost: {tc_l8}")
    print(f"l70 cost: {tc_l70}")
    print(f"hllm cost: {tc_hllm}")
    print(f"hllm l8 %: {hllm_l8 / l}")
    print(f"hllm l70 %: {l70_pred / l}")

    print("\nHybridLLM Metrics:")
    print(f"This is the threshold: {threshold}")
    print(f"l8 pred count: {l8_pred}")
    print(f"l70 pred count: {l70_pred}")

    # print(f"l8 prec: {safe_div(l8_pred_cor, l8_pred)}")
    # print(f"l8 rec: {safe_div(l8_pred_cor, l8_real)}")
    # print(f"l70 prec: {safe_div(l70_pred_cor, l70_pred)}")
    # print(f"l70 rec: {safe_div(l70_pred_cor, l70_real)}")

    # print("\nFailure counts:")
    # print(f"target fails: {fail_target}")
    # print(f"l8 fails: {fail_l8}")
    # print(f"l70 fails: {fail_l70}")

    # print(f"\nSaved routed outputs to: {out_path}")