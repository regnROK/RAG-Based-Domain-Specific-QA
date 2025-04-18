# Financial Question-Answering System with Fine-tuned Flan-T5-XL

This project implements a Retrieval-Augmented Generation (RAG) system for answering questions on financial documents. It uses a fine-tuned version of Google's Flan-T5-XL language model, specifically trained on the TAT-QA (Tabular and Textual Question Answering) dataset to better handle financial and tabular data.

## Features

- **Fine-tuned Language Model**: Uses Flan-T5-XL fine-tuned on financial QA datasets for more accurate answers
- **Retrieval-Augmented Generation**: Leverages vector similarity search to find relevant contexts from documents
- **PDF Processing**: Extract text from PDFs and process them for question answering
- **Interactive Web UI**: User-friendly Streamlit interface for uploading documents and asking questions
- **Comprehensive Evaluation**: Includes tools to evaluate model performance with various metrics

## Project Structure

- `app.py`: Streamlit web application for interacting with the QA system
- `model.py`: Training code for fine-tuning the Flan-T5-XL model on TAT-QA dataset
- `rag.py`: Implementation of the Retrieval-Augmented Generation pipeline
- `testing.py`: Script for testing the fine-tuned model
- `evaluation.py`: Comprehensive evaluation pipeline comparing original and fine-tuned models
- `hf_push.py`: Utility to push the fine-tuned model to Hugging Face Hub

## Installation

1. Clone this repository:
```bash
git clone https://github.com/regnROK/RAG-Based-Domain-Specific-QA
cd financial-qa-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Application

```bash
streamlit run app.py
```

This opens a web interface where you can:
1. Upload a PDF document
2. Ask questions about the document
3. Get AI-generated answers based on the document content
4. View model evaluation results

### Testing the Model

```bash
python testing.py
```

This will run a few test questions against the fine-tuned model to verify it's working correctly.

### Evaluating the Model

```bash
python evaluation.py
```

This runs a comprehensive evaluation comparing the original Flan-T5-XL model with the fine-tuned version on the TAT-QA test set, generating:
- Detailed metrics (ROUGE, BLEU, F1, Exact Match, Edit Similarity, BERT Similarity)
- Visualizations of performance differences
- CSV file with all evaluation results

## Model Training

The model was fine-tuned on the TAT-QA dataset, which contains questions and answers from financial tables and text. The training process:

1. Processes the TAT-QA dataset, combining textual and tabular data
2. Fine-tunes the Flan-T5-XL model using the processed data
3. Evaluates on a validation split to monitor performance
4. Saves the best model checkpoints

To train the model yourself:

```bash
python model.py
```

Note: Training requires significant computational resources. The provided code includes optimizations for memory efficiency, but you may need to adjust batch sizes and other parameters based on your hardware.

## Model Deployment

The fine-tuned model is available on Hugging Face Hub:
- Model ID: `base/flan-t5-xl_finance_QA`

To push updates to the model:

```bash
export HF_TOKEN=your_huggingface_token
python hf_push.py
```

## Vector Database

The project uses ChromaDB as a vector database for storing document embeddings. The database is created and populated when documents are uploaded through the web interface.

## Evaluation Results

The model evaluation compares the fine-tuned model against the original Flan-T5-XL base model using multiple metrics:

- ROUGE-L: Measures longest common subsequence
- BLEU: Measures precision of n-grams
- F1 Score: Harmonic mean of precision and recall
- Exact Match: Binary indicator of perfect answers
- Edit Similarity: Measure of string edit distance ratio
- BERT Similarity: Semantic similarity using BERT embeddings

Visualizations of these metrics are generated in the `/output` directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- This project uses the TAT-QA dataset from Microsoft Research
- Fine-tuned models are based on Google's Flan-T5-XL architecture
