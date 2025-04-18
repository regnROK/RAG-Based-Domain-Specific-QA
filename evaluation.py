import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
import re

# Set device
device = torch.device("cpu")

# Load models
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model_original = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl").to(device)
model_finetuned = AutoModelForSeq2SeqLM.from_pretrained("base/flan-t5-xl_finance_QA").to(device)

# Load Sentence-BERT model for semantic similarity
bert_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# Ensure charts directory exists
os.makedirs("output", exist_ok=True)

def normalize_text(text):
    if not isinstance(text, str):
        return "UNKNOWN"
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\.\,\-\$%]", "", text)  # keep $, %, ., etc.
    return text

def preprocess_dataset(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    rows = []
    for item in data:
        context = " ".join([p["text"] for p in item.get("paragraphs", [])])
        for q in item.get("questions", []):
            question = q.get("question", "UNKNOWN")
            answers = q.get("answer", [])
            if isinstance(answers, list) and answers and isinstance(answers[0], dict):
                answer_text = answers[0].get("answer_text", "UNKNOWN")
            else:
                answer_text = "UNKNOWN"
            rows.append({
                "Question": question,
                "Context": context,
                "Expected Answer": answer_text
            })
            
    return pd.DataFrame(rows)

def compute_metrics(expected, predicted):
    expected_norm = normalize_text(expected)
    predicted_norm = normalize_text(predicted)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = scorer.score(expected_norm, predicted_norm)["rougeL"].fmeasure

    bleu = sentence_bleu(
        [expected_norm.split()],
        predicted_norm.split(),
        smoothing_function=SmoothingFunction().method1
    )

    expected_tokens = expected_norm.split()
    predicted_tokens = predicted_norm.split()
    common = set(expected_tokens) & set(predicted_tokens)
    num_same = len(common)

    precision = num_same / len(predicted_tokens) if predicted_tokens else 0
    recall = num_same / len(expected_tokens) if expected_tokens else 0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0

    em = 1.0 if expected_norm == predicted_norm else 0.0
    edit_distance = SequenceMatcher(None, expected_norm, predicted_norm).ratio()

    # BERT Similarity
    emb1 = bert_model.encode(expected_norm, convert_to_tensor=True)
    emb2 = bert_model.encode(predicted_norm, convert_to_tensor=True)
    bert_sim = util.pytorch_cos_sim(emb1, emb2).item()

    return rouge, bleu, f1, em, edit_distance, bert_sim

def evaluate_model(df, model, tokenizer, label):
    outputs = []
    
    # Adding a progress bar for evaluation
    with tqdm(total=len(df), desc=f"Evaluating {label} Model", unit="question") as pbar:
        for _, row in df.iterrows():
            question = row["Question"]
            context = row["Context"]
            expected_answer = row["Expected Answer"]

            input_text = f"question: {question} context: {context}"
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

            with torch.no_grad():
                output = model.generate(**inputs, max_length=128)
            predicted = tokenizer.decode(output[0], skip_special_tokens=True)

            rouge, bleu, f1, em, edit_sim, bert_sim = compute_metrics(expected_answer, predicted)

            outputs.append({
                "Question": question,
                "Context": context,
                "Expected Answer": expected_answer,
                f"{label} Prediction": predicted,
                f"ROUGE-L ({label})": rouge,
                f"BLEU ({label})": bleu,
                f"F1 Score ({label})": f1,
                f"Exact Match ({label})": em,
                f"Edit Similarity ({label})": edit_sim,
                f"BERT Similarity ({label})": bert_sim
            })

            # Update progress bar after each evaluation
            pbar.update(1)

    return pd.DataFrame(outputs)

def merge_results(df_orig, df_finetuned):
    return pd.concat([
        df_orig.drop(columns=["Context", "Expected Answer"]),
        df_finetuned.drop(columns=["Context", "Expected Answer", "Question"])
    ], axis=1)

def generate_visuals(df):
    metrics = ["ROUGE-L", "BLEU", "F1 Score", "Exact Match", "Edit Similarity", "BERT Similarity"]
    for metric in metrics:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[f"{metric} (Original)"], color="red", label="Original", kde=True, stat="density")
        sns.histplot(df[f"{metric} (Fine-Tuned)"], color="green", label="Fine-Tuned", kde=True, stat="density")
        plt.title(f"{metric} Score Distribution")
        plt.xlabel(f"{metric} Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"output/{metric.lower().replace(' ', '_')}_distribution.png")
        plt.close()

        plt.figure(figsize=(14, 6))
        melted_df = df[["Question", f"{metric} (Original)", f"{metric} (Fine-Tuned)"]].melt(
            id_vars="Question", var_name="Model", value_name=metric)
        sns.barplot(x="Question", y=metric, hue="Model", data=melted_df)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{metric} Comparison per Question")
        plt.tight_layout()
        plt.savefig(f"output/{metric.lower().replace(' ', '_')}_bar_comparison.png")
        plt.close()

# Main Pipeline
if __name__ == "__main__":
    # Preprocess dataset
    df_data = preprocess_dataset("Data/TAT-QA/dataset_raw/tatqa_dataset_test_gold.json")

    # Evaluate both original and fine-tuned models with progress bars
    df_original = evaluate_model(df_data, model_original, tokenizer, "Original")
    df_finetuned = evaluate_model(df_data, model_finetuned, tokenizer, "Fine-Tuned")

    # Merge results from both models
    df_combined = merge_results(df_original, df_finetuned)

    # Save results
    df_combined.to_csv("evaluation.csv", index=False)
    print("Results saved to evaluation.csv")

    # Generate visuals and save charts
    generate_visuals(df_combined)
    print("Charts saved in /output")

