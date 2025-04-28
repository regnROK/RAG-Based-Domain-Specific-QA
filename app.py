import streamlit as st
import rag
import pandas as pd
import base64

st.set_page_config(page_title="RAG-based PDF QA", layout="wide")
st.title("RAG-based QA with Fine-Tuned Flan-T5-XL")

@st.cache_resource(show_spinner="Loading fine-tuned model...")
def get_finetuned_model():
    return rag.load_finetuned_model()

# Load only once
tokenizer, model = get_finetuned_model()

# Sidebar Navigation
st.sidebar.title("Navigation")
st.markdown("""
    <style>
    div[data-baseweb="select"] > div {
        box-shadow: none !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px;
        cursor: pointer;
    }
    div[data-baseweb="select"] > div:focus-within {
        outline: none !important;
        box-shadow: none !important;
    }
    div[data-baseweb="select"] input {
        caret-color: transparent !important;
    }
    </style>
""", unsafe_allow_html=True)

mode = st.sidebar.selectbox("Select Mode", ["Inference", "Evaluation"])

def render_image(path, caption, height=400):
    try:
        with open(path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
            st.markdown(
                f"""
                <div style="text-align:center; margin-bottom: 1rem;">
                    <img src="data:image/png;base64,{encoded}" 
                         style="height: {height}px; border-radius: 8px; border: 1px solid #ccc;" />
                    <p style="font-size: 14px; color: #555;">{caption}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.warning(f"Could not load image {path}: {e}")

if mode == "Inference":
    st.subheader("PDF QA Inference")

    # --- Upload Section ---
    st.markdown("### 1. Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file:
        st.info("Processing PDF...")
        pdf_text = rag.extract_text_from_pdf(uploaded_file)
        text_chunks = rag.chunk_text(pdf_text)
        rag.embed_and_store_chunks(text_chunks)
        st.success("PDF processed and stored in the vector database.")

    # --- Question Input Section ---
    st.markdown("### 2. Ask a Question")
    question = st.text_input("Enter your question about the uploaded document:")

    # --- Answer Output Section ---
    if st.button("Get Answer", use_container_width=True) and question:
        context = rag.retrieve_relevant_chunks(question)
        if context:
            answer = rag.generate_answer(
                question, context, model_type="finetuned",
                tokenizer=tokenizer, model=model
            )
            st.markdown("### 3. Answer")
            st.text_area("Generated Answer", value=answer, height=180, disabled=True)
        else:
            st.warning("No relevant information found.")

elif mode == "Evaluation":
    st.subheader("Model Evaluation on Preloaded Results")

    try:
        # Load full evaluation CSV
        eval_results = pd.read_csv("output/evaluation.csv")

        # Select relevant columns
        selected_columns = ["Question", "Original Prediction", "Fine-Tuned Prediction",
                            "BERT Similarity (Original)", "BERT Similarity (Fine-Tuned)",
                            "Edit Similarity (Original)", "Edit Similarity (Fine-Tuned)"]

        # Get top 5 rows with highest BERT similarity (Fine-Tuned)
        top_bert = eval_results.nlargest(10, "BERT Similarity (Fine-Tuned)")[selected_columns]

        # Get top 5 rows with highest Edit similarity (Fine-Tuned)
        top_edit = eval_results.nlargest(10, "Edit Similarity (Fine-Tuned)")[selected_columns]

        # Combine and drop duplicates (in case some overlap)
        best_results = pd.concat([top_bert, top_edit]).drop_duplicates().reset_index(drop=True)

        # Display best results
        st.write("### Results from Fine-Tuned Model")
        st.dataframe(best_results)

        # Calculate and display average scores
        avg_bert_finetuned = eval_results["BERT Similarity (Fine-Tuned)"].mean()
        avg_edit_finetuned = eval_results["Edit Similarity (Fine-Tuned)"].mean()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg BERT Similarity (Fine-Tuned)", round(avg_bert_finetuned, 4))
        with col2:
            st.metric("Avg Edit Similarity (Fine-Tuned)", round(avg_edit_finetuned, 4))

        # Preview first few rows
        st.write("### Preview of Evaluation Results")
        limited_results = eval_results[selected_columns].head(10)
        st.dataframe(limited_results)

        # Show evaluation graphs
        st.write("### BERT Similarity Comparison")
        col3, col4 = st.columns(2)
        with col3:
            st.image("output/bert_similarity_bar_comparison.png", 
                     caption="BERT Similarity - Original vs Fine-Tuned (Bar Comparison)", 
                     use_container_width=True)
        with col4:
            st.image("output/bert_similarity_distribution.png", 
                     caption="BERT Similarity Score Distribution", 
                     use_container_width=True)

        st.write("### Edit Similarity Comparison")
        col5, col6 = st.columns(2)
        with col5:
            st.image("output/edit_similarity_bar_comparison.png", 
                     caption="Edit Similarity - Original vs Fine-Tuned (Bar Comparison)", 
                     use_container_width=True)
        with col6:
            st.image("output/edit_similarity_distribution.png", 
                     caption="Edit Similarity Score Distribution", 
                     use_container_width=True)

    except Exception as e:
        st.error(f"Error loading evaluation file or images: {e}")
