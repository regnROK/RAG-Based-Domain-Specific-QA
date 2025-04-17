# import json
# from datasets import Dataset
# from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
# import torch

# # Check for available GPUs
# device = torch.device("cpu")

# # 1. Load and preprocess the TAT-QA dataset
# def load_tatqa_dataset(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     inputs = []
#     targets = []
    
#     for item in data:
#         # Combine paragraphs into a single string
#         paragraphs = " ".join([p["text"] for p in item["paragraphs"]])
#         # Flatten table into a string (simplified approach)
#         table_rows = [" ".join(row) for row in item["table"]["table"]]
#         table = " ".join(table_rows)
#         context = f"{paragraphs} {table}"
        
#         # Iterate over each question in the "questions" list
#         for question_entry in item["questions"]:
#             question = question_entry["question"]
#             answer = question_entry["answer"]
            
#             # Handle multi-span answers (list) by joining them into a string
#             if isinstance(answer, list):
#                 answer_text = ", ".join(str(a) for a in answer)
#             else:
#                 answer_text = str(answer)
            
#             input_text = f"question: {question} context: {context}"
#             targets.append(answer_text)
#             inputs.append(input_text)
    
#     return Dataset.from_dict({"input_text": inputs, "target_text": targets})

# # Load train and validation datasets
# train_dataset = load_tatqa_dataset("Data/TAT-QA/dataset_raw/tatqa_dataset_train.json")
# val_dataset = load_tatqa_dataset("Data/TAT-QA/dataset_raw/tatqa_dataset_dev.json")

# # 2. Initialize tokenizer and model
# model_name = "google/flan-t5-xl"
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name).to("cpu")

# # Use all available GPUs with DeepSpeed if possible
# # if torch.cuda.device_count() > 1:
# #     print(f"Using {torch.cuda.device_count()} GPUs")
# #     model = torch.nn.DataParallel(model)

# # model.to(device)
# # model.gradient_checkpointing_enable()  # Reduce memory usage

# # 3. Tokenize the dataset
# def preprocess_function(examples):
#     inputs = tokenizer(examples["input_text"], max_length=512, truncation=True, padding="max_length")
#     targets = tokenizer(examples["target_text"], max_length=128, truncation=True, padding="max_length")
    
#     inputs["labels"] = targets["input_ids"]
#     return inputs

# train_dataset = train_dataset.map(preprocess_function, batched=True)
# val_dataset = val_dataset.map(preprocess_function, batched=True)

# # 4. Set training arguments with memory-efficient options
# training_args = TrainingArguments(
#     output_dir="./flan-t5-xl-finetuned-tatqa",
#     evaluation_strategy="steps",
#     learning_rate=2e-5,
#     eval_steps=100,
#     per_device_train_batch_size=1,  # Keep minimal due to limited GPU memory
#     per_device_eval_batch_size=1,
#     max_steps=1000,
#     weight_decay=0.01,
#     save_strategy="steps",
#     load_best_model_at_end=True,
#     logging_dir="./logs",
#     logging_steps=10,
#     use_cpu=True,
#     # fp16=True,  # Mixed precision for faster training
#     # dataloader_num_workers=4,
#     # report_to=["tensorboard"],
#     save_total_limit=2,
#     # gradient_accumulation_steps=8,  # Compensate for low batch size
#     # optim="adamw_torch",
#     # ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
#     # deepspeed="ds_config.json"  # Enable DeepSpeed optimization
# )

# # 5. Initialize Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     tokenizer=tokenizer,
# )

# # 6. Fine-tune the model
# trainer.train()

# # 7. Save the fine-tuned model
# model.module.save_pretrained("./flan-t5-xl-finetuned-tatqa/final_model") if isinstance(model, torch.nn.DataParallel) else model.save_pretrained("./flan-t5-xl-finetuned-tatqa/final_model")
# tokenizer.save_pretrained("./flan-t5-xl-finetuned-tatqa/final_model")

# # 8. Example inference
# def generate_answer(question, context):
#     input_text = f"question: {question} context: {context}"
#     inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
#     outputs = model.generate(**inputs, max_length=128)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Test the model
# sample_question = "What does the Weighted average actuarial assumptions consist of?"
# sample_context = "Actuarial assumptions The Group’s scheme liabilities are measured using the projected unit credit method using the principal actuarial assumptions set out below: Rate of inflation 2.9 2.9 3.0 Rate of increase in salaries 2.7 2.7 2.6 Discount rate 2.3 2.5 2.6"
# print(generate_answer(sample_question, sample_context))



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
