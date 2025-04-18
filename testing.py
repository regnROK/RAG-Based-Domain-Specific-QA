# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("flan-t5-xl-finetuned-tatqa/final_model")
# model = AutoModelForSeq2SeqLM.from_pretrained("flan-t5-xl-finetuned-tatqa/final_model")

# print("MODEL LOADED!")

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model(model_path):
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    return tokenizer, model, device

def generate_answer(tokenizer, model, device, question, context, max_length=128):
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load the fine-tuned model
tokenizer, model, device = load_model("base/flan-t5-xl_finance_QA")

# Example test cases
test_cases = [
    {"question": "What does the Weighted average actuarial assumptions consist of?", 
     "context": "Actuarial assumptions The Group’s scheme liabilities are measured using the projected unit credit method using the principal actuarial assumptions set out below: Rate of inflation 2.9 2.9 3.0 Rate of increase in salaries 2.7 2.7 2.6 Discount rate 2.3 2.5 2.6"},
    
    {"question": "What is the net revenue for the year 2023?", 
     "context": "The company's financial report states that the net revenue for 2023 was $5.4 million, with a gross margin of 45%."}
]

# Run inference on test cases
for test in test_cases:
    answer = generate_answer(tokenizer, model, device, test["question"], test["context"])
    print(f"Question: {test['question']}")
    print(f"Generated Answer: {answer}\n")
