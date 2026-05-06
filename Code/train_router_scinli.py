import os
import random
import numpy as np
import pandas as pd
import torch
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


# =========================
# 1. Reproducibility
# =========================
SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# =========================
# 2. User settings
# =========================
MODEL_NAME = "microsoft/deberta-v3-large"  
SENT1_COL = "Sentence 1"
SENT2_COL = "Sentence 2"
LABEL_COL = "full_lean_l8" # corresponds to l8 >= l70 - label is 1 (route to Llama 8B) for all ties (1 and 1) and (0 and 0)

OUTPUT_DIR = "./router_deberta_v3_large_SciNLI"
df = pd.read_csv("/home/tmalik6/LLMR/Code/hllm/training_data/HLLM_train_SciNLI.csv")
MAX_LENGTH = 512
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 5
WARMUP_RATIO = 0.1
FP16 = torch.cuda.is_available()



# =========================
# 3. Data
# =========================

# Labeled dataset - created from GSM8K and MATH train - currently concatenated (GSM8K first)

df["stratify_key"] = df["full_lean_l8"].astype(str)

# =========================
# 4. Train/val split
# =========================
def make_splits(df: pd.DataFrame):
    # Stratified split:
    # 80% train, 20% validation
    
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["stratify_key"]
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# =========================
# 5. Tokenization
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    return tokenizer(
        examples[SENT1_COL],
        examples[SENT2_COL],
        truncation=True,
        max_length=MAX_LENGTH,
    )


# =========================
# 6. Metrics
# =========================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred):
    """
    For num_labels=1, model outputs one logit per example.
    We threshold sigmoid(logit) at 0.5 for binary predictions.
    """
    logits, labels = eval_pred

    # logits may come as shape (N, 1); flatten it
    logits = np.squeeze(logits)
    probs = sigmoid(logits)
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    # ROC-AUC only works if both classes are present
    unique_labels = np.unique(labels)
    if len(unique_labels) == 2:
        metrics["roc_auc"] = roc_auc_score(labels, probs)

    return metrics


# =========================
# 7. Custom Trainer for BCEWithLogitsLoss
# =========================

class BinaryClassificationTrainer(Trainer):
    """
    Hugging Face sometimes handles regression-like setups a bit differently when num_labels=1.
    To make the loss explicit and stable, we define BCEWithLogitsLoss ourselves.
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)

        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# =========================
# 8. Main training function
# =========================
def train_router(df: pd.DataFrame):

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main_process = local_rank == 0
    

    train_df, val_df = make_splits(df)

    if is_main_process:
        print("Split sizes:")
        print(f"Train: {len(train_df)}")
        print(f"Val:   {len(val_df)}")

        print("\nLabel balance:")
        print("Train:\n", train_df[LABEL_COL].value_counts(normalize=True))
        print("Val:\n", val_df[LABEL_COL].value_counts(normalize=True))
    

    # Convert pandas -> HF Dataset
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    

    # Rename label column to what Trainer expects
    train_ds = train_ds.rename_column(LABEL_COL, "labels")
    val_ds = val_ds.rename_column(LABEL_COL, "labels")
    

    # Tokenize
    train_ds = train_ds.map(tokenize_function, batched=True)
    val_ds = val_ds.map(tokenize_function, batched=True)
    

    # Keep only needed columns
    keep_cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in train_ds.column_names:
        keep_cols.append("token_type_ids")

    train_ds.set_format(type="torch", columns=keep_cols)
    val_ds.set_format(type="torch", columns=keep_cols)
    

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # num_labels=1 -> single logit
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=FP16,
        bf16=False,
        report_to="none",
        save_total_limit=3,
        ddp_find_unused_parameters=False,
    )

    trainer = BinaryClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if is_main_process:
        print("\nStarting training...")
    trainer.train()

    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    if is_main_process:
        print("\nValidation results:")
        print(val_metrics)
    
    print(val_metrics)


    # Save best model + tokenizer
    best_dir = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    print(f"\nSaved best model to: {best_dir}")

    return trainer, val_ds, val_metrics





# =========================
# 10. Example usage
# =========================

def main():

    train_router(df)

if __name__ == "__main__":
    main()