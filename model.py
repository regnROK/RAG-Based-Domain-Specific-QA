import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import train_test_split

# Check for available GPUs
device = torch.device("cpu")

# 1. Load and preprocess the TAT-QA dataset
def load_tatqa_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    inputs = []
    targets = []
    
    for item in data:
        paragraphs = " ".join([p["text"] for p in item["paragraphs"]])
        table_rows = [" ".join(row) for row in item["table"]["table"]]
        table = " ".join(table_rows)
        context = f"{paragraphs} {table}"
        
        for question_entry in item["questions"]:
            question = question_entry["question"]
            answer = question_entry["answer"]
            
            if isinstance(answer, list):
                answer_text = ", ".join(str(a) for a in answer)
            else:
                answer_text = str(answer)
            
            input_text = f"question: {question} context: {context}"
            targets.append(answer_text)
            inputs.append(input_text)
    
    return pd.DataFrame({"Question": inputs, "Answer": targets})

# Load dataset
dataset = load_tatqa_dataset("Data/TAT-QA/dataset_raw/tatqa_dataset_train.json")
eval_dataset = load_tatqa_dataset("Data/TAT-QA/dataset_raw/tatqa_dataset_dev.json")
train_data, test_data = train_test_split(dataset, test_size=0.2)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

def preprocess_for_t5(dataset, max_input_length=512, max_output_length=128):
    inputs = dataset["Question"].tolist()
    targets = dataset["Answer"].tolist()
    
    tokenized_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    tokenized_targets = tokenizer(
        targets,
        max_length=max_output_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return tokenized_inputs, tokenized_targets

tokenized_train_inputs, tokenized_train_targets = preprocess_for_t5(train_data)
tokenized_eval_inputs, tokenized_eval_targets = preprocess_for_t5(eval_dataset)
tokenized_test_inputs, tokenized_test_targets = preprocess_for_t5(test_data)

# Custom Dataset
class MedicalQADataset(TorchDataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs["input_ids"][idx],
            "attention_mask": self.inputs["attention_mask"][idx],
            "labels": self.targets["input_ids"][idx],
        }

train_dataset = MedicalQADataset(tokenized_train_inputs, tokenized_train_targets)
eval_dataset = MedicalQADataset(tokenized_eval_inputs, tokenized_eval_targets)
test_dataset = MedicalQADataset(tokenized_test_inputs, tokenized_test_targets)

# Load Model
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl").to(device)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./flan-t5-xl-finetuned-tatqa",
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=100,
    logging_dir="./logs",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    max_steps=1000,
    # num_train_epochs=2,
    learning_rate=2e-5,
    save_steps=250,
    save_total_limit=4,
    weight_decay=0.01,
    logging_steps=50,
    # warmup_steps=200,
    use_cpu = True
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

# Train Model
trainer.train()

# Evaluate Model
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# Test Model
test_results = trainer.evaluate(test_dataset)
print("Test Results:", test_results)

# Save Model
model.save_pretrained("./flan-t5-xl-finetuned-tatqa/final_model")
tokenizer.save_pretrained("./flan-t5-xl-finetuned-tatqa/final_model")
