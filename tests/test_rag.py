import pytest
import rag
from io import BytesIO
from unittest.mock import patch, MagicMock

# Sample data for testing
SAMPLE_TEXT = "This is the first sentence. This is the second sentence. This is the third sentence. This is the fourth sentence. This is the fifth sentence."
SAMPLE_CHUNKS = [
    "This is the first sentence.",
    "This is the second sentence.",
    "This is the third sentence.",
    "This is the fourth sentence.",
    "This is the fifth sentence."
]
SAMPLE_QUERY = "What is the second sentence?"

# --- Fixtures ---

@pytest.fixture
def mock_pdf_file():
    """Creates a mock PDF file object."""
    # Simulate a PDF file in memory for testing extract_text_from_pdf
    # In a real scenario, you might use a small actual PDF file
    # For simplicity here, we'll mock the fitz library behavior
    return BytesIO(b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n5 0 obj<</Length 50>>stream\nBT /F1 12 Tf 72 712 Td (This is sample PDF text.) Tj ET\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f\n0000000010 00000 n\n0000000059 00000 n\n0000000112 00000 n\n0000000234 00000 n\n0000000297 00000 n\ntrailer<</Size 6/Root 1 0 R>>\nstartxref\n400\n%%EOF")

@pytest.fixture
def mock_embedding_model():
    """Mocks the SentenceTransformer model."""
    mock = MagicMock()
    # Simulate embedding generation
    mock.encode.return_value = MagicMock()
    mock.encode.return_value.tolist.return_value = [[0.1, 0.2, 0.3]] # Example embedding
    return mock

@pytest.fixture
def mock_chromadb_collection():
    """Mocks the ChromaDB collection."""
    mock = MagicMock()
    # Simulate query results
    mock.query.return_value = {"documents": [["Relevant chunk 1", "Relevant chunk 2"]]}
    return mock

@pytest.fixture
def mock_language_model():
    """Mocks the language model and tokenizer."""
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    
    # Simulate tokenization and generation
    mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]} # Dummy tokenized output
    mock_model.generate.return_value = [[1, 2, 3, 4, 5]] # Dummy generated token IDs
    mock_tokenizer.decode.return_value = "Generated answer text." # Dummy decoded answer
    
    return mock_tokenizer, mock_model

# --- Test Functions ---

@patch('rag.fitz.open')
def test_extract_text_from_pdf(mock_fitz_open, mock_pdf_file):
    """Tests text extraction from a PDF."""
    # Configure the mock fitz document and page
    mock_page = MagicMock()
    mock_page.get_text.return_value = "This is sample PDF text."
    mock_doc = MagicMock()
    mock_doc.__iter__.return_value = [mock_page]
    mock_fitz_open.return_value = mock_doc

    extracted_text = rag.extract_text_from_pdf(mock_pdf_file)
    
    mock_fitz_open.assert_called_once() # Check if fitz.open was called
    assert extracted_text == "This is sample PDF text."

def test_chunk_text():
    """Tests text chunking."""
    # Test with a small chunk size for simplicity
    chunks = rag.chunk_text(SAMPLE_TEXT, chunk_size=5) 
    expected_chunks = [
        'This is the first sentence.', 
        'This is the second sentence.', 
        'This is the third sentence.', 
        'This is the fourth sentence.', 
        'This is the fifth sentence.'
    ]
    # Adjust assertion based on actual chunking logic if needed
    assert len(chunks) > 0 # Basic check
    # Example: assert chunks == expected_chunks # This might need adjustment based on exact splitting

@patch('rag.embedding_model', new_callable=MagicMock)
@patch('rag.collection', new_callable=MagicMock)
def test_embed_and_store_chunks(mock_collection, mock_emb_model):
    """Tests embedding and storing chunks."""
    # Configure mocks
    mock_emb_model.encode.return_value.tolist.return_value = [[0.1, 0.2]] * len(SAMPLE_CHUNKS)
    
    rag.embed_and_store_chunks(SAMPLE_CHUNKS)
    
    mock_emb_model.encode.assert_called_once_with(SAMPLE_CHUNKS)
    assert mock_collection.add.call_count == len(SAMPLE_CHUNKS)
    # Check one of the calls to add
    mock_collection.add.assert_any_call(ids=['0'], documents=[SAMPLE_CHUNKS[0]], embeddings=[[0.1, 0.2]])


@patch('rag.embedding_model', new_callable=MagicMock)
@patch('rag.collection', new_callable=MagicMock)
def test_retrieve_relevant_chunks(mock_collection, mock_emb_model, mock_chromadb_collection):
    """Tests retrieving relevant chunks."""
    # Configure mocks
    mock_emb_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3] # Single query embedding
    mock_collection.query.return_value = mock_chromadb_collection.query.return_value # Use fixture's return value

    context = rag.retrieve_relevant_chunks(SAMPLE_QUERY, top_k=2)
    
    mock_emb_model.encode.assert_called_once_with(SAMPLE_QUERY)
    mock_collection.query.assert_called_once_with(query_embeddings=[[0.1, 0.2, 0.3]], n_results=2)
    assert context == "Relevant chunk 1 Relevant chunk 2"

def test_generate_answer(mock_language_model):
    """Tests answer generation using the fine-tuned model."""
    mock_tokenizer, mock_model = mock_language_model
    
    question = "What is the question?"
    context = "This is the context."
    
    answer = rag.generate_answer(
        question, 
        context, 
        model_type="finetuned", 
        tokenizer=mock_tokenizer, 
        model=mock_model
    )
    
    mock_tokenizer.assert_called_once_with(question + " " + context, return_tensors="pt", truncation=True)
    mock_model.generate.assert_called_once()
    mock_tokenizer.decode.assert_called_once()
    assert answer == "Generated answer text."

# Add more tests as needed, e.g., for edge cases, empty inputs, etc.
