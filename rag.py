import fitz
import torch
import chromadb
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional, IO

# Load models
device = torch.device("cpu")
# tokenizer = AutoTokenizer.from_pretrained("anjalipatel03/flan-t5-xl_finance_QA")
# model = AutoModelForSeq2SeqLM.from_pretrained("anjalipatel03/flan-t5-xl_finance_QA").to(device)
embedding_model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")

def load_finetuned_model() -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    tokenizer = AutoTokenizer.from_pretrained("base/flan-t5-xl_finance_QA")
    model = AutoModelForSeq2SeqLM.from_pretrained("base/flan-t5-xl_finance_QA")
    return tokenizer, model

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_qa")

def extract_text_from_pdf(pdf_file: IO[bytes]) -> str:
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join([page.get_text("text") for page in doc])

def chunk_text(text: str, chunk_size: int = 512) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_and_store_chunks(chunks: List[str]) -> None:
    embeddings = embedding_model.encode(chunks).tolist()
    for i, chunk in enumerate(chunks):
        collection.add(ids=[str(i)], documents=[chunk], embeddings=[embeddings[i]])

def retrieve_relevant_chunks(query: str, top_k: int = 3) -> str:
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return " ".join(results["documents"][0]) if results["documents"] else ""

def generate_answer(
    question: str,
    context: str,
    model_type: str = "finetuned",
    tokenizer: Optional[AutoTokenizer] = None,
    model: Optional[AutoModelForSeq2SeqLM] = None
) -> str:
    if model_type == "finetuned" and model and tokenizer:
        # Tokenize and generate with fine-tuned model
        inputs = tokenizer(question + " " + context, return_tensors="pt", truncation=True)
        output = model.generate(**inputs)
        return tokenizer.decode(output[0], skip_special_tokens=True)

