import streamlit as st
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA, load_summarize_chain
from langchain.prompts import PromptTemplate

# --- 1. Streamlit UI Configuration ---
st.set_page_config(page_title="Llama Text Agent 📝", layout="wide")
st.title("📄 Llama-Powered Text Reading and Analysis Agent")

with st.sidebar:
    st.header("⚙️ Configuration")
    # SURE: Allowing model selection is a good practice. `llama3:8b` is a powerful and efficient model
    # suitable for CPU-only systems.
    selected_model = st.selectbox(
        "Select a Model",
        ["llama2", "phi3:mini"],
        index=0,
        help="Ensure the selected model is downloaded in Ollama."
    )
    # SURE: Chunk size and overlap are critical for context engineering.
    # These values are a good starting point for balancing context quality and performance.
    chunk_size = st.slider(
        "Text Chunk Size", 500, 2000, 1000,
        help="Size of text chunks for processing. Smaller chunks are faster but may lose context."
    )
    chunk_overlap = st.slider(
        "Text Chunk Overlap", 0, 500, 200,
        help="Overlap between text chunks to maintain context continuity."
    )

# --- 2. Model and Embeddings Initialization ---
@st.cache_resource
def initialize_components(model_name):
    """
    Initializes the LLM and embeddings model.
    Using @st.cache_resource to avoid reloading models on each interaction.
    """
    # SURE: This is the standard and effective way to initialize the Ollama LLM and embeddings within LangChain.
    llm = Ollama(model=model_name, base_url="http://localhost:11434")
    embeddings = OllamaEmbeddings(model=model_name, base_url="http://localhost:11434")
    return llm, embeddings

llm, embeddings = initialize_components(selected_model)

# --- 3. Core Logic: Prompt & Context Engineering ---

# Prompt for Summarization
# SURE: This map-reduce prompt strategy is robust for summarizing large documents.
# It first creates a summary of each chunk (map) and then combines them into a final summary (reduce).
summarize_prompt_template = """
Write a concise summary of the following text:
---
{text}
---
CONCISE SUMMARY:
"""
summarize_prompt = PromptTemplate(template=summarize_prompt_template, input_variables=["text"])

# Prompt for Question-Answering
# SURE: This prompt is engineered to prevent hallucination by forcing the model to rely only on the provided text.
# The instruction to state when an answer is unknown is a key technique for factual accuracy.
qa_prompt_template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Context:
---
{context}
---

Question: {question}

Answer:
"""
qa_prompt = PromptTemplate(template=qa_prompt_template, input_variables=["context", "question"])

# --- 4. Application Logic and UI ---
st.header("1. Upload Your Text Document")
uploaded_file = st.file_uploader(
    "Upload a .txt file", type=["txt"],
    help="The agent will read, summarize, and answer questions based on this document."
)

if uploaded_file is not None:
    # Read and process the document
    try:
        # SURE: Reading and splitting text this way is standard practice for handling large documents
        # that would otherwise exceed the model's context window.
        raw_text = uploaded_file.read().decode("utf-8")
        st.info("File successfully uploaded and read.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        texts = text_splitter.split_text(raw_text)

        # Create a vector store for RAG (Retrieval-Augmented Generation)
        @st.cache_data
        def create_vector_store(_texts):
            with st.spinner("Creating vector store for Q&A...                       ...this may take sometime"):
                vectorstore = FAISS.from_texts(texts=_texts, embedding=embeddings)
            return vectorstore

        vectorstore = create_vector_store(texts)
        st.success("Vector store created...Ready for summarization and Q&A ")

        # Display tabs for different actions
        tab1, tab2 = st.tabs(["📝 Summarize Document", "❓ Ask a Question"])

        with tab1:
            st.subheader("Summarize the Document")
            if st.button("Generate Summary"):
                with st.spinner(f"Generating summary with {selected_model}..."):
                    # SURE: `map_reduce` is the correct chain type for summarizing large documents chunk by chunk.
                    docs = text_splitter.create_documents(texts)
                    summary_chain = load_summarize_chain(
                        llm,
                        chain_type="map_reduce",
                        map_prompt=summarize_prompt,
                        combine_prompt=summarize_prompt,
                        verbose=False
                    )
                    summary = summary_chain.run(docs)
                    st.markdown("### Summary:")
                    st.write(summary)

        with tab2:
            st.subheader("Ask a Question Based on the Document")
            user_question = st.text_input("Enter your question here:")
            if st.button("Get Answer"):
                if user_question:
                    with st.spinner(f"Searching for the answer with {selected_model}..."):
                        # SURE: This `RetrievalQA` chain combines retrieval from the vector store with generation from the LLM.
                        # The `chain_type_kwargs` injects our custom, safer prompt.
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=vectorstore.as_retriever(),
                            chain_type_kwargs={"prompt": qa_prompt},
                            return_source_documents=True
                        )
                        result = qa_chain.invoke({"query": user_question}) # Using invoke for newer LangChain versions
                        st.markdown("### Answer:")
                        st.write(result["result"])

                        with st.expander("Show source context"):
                            st.write(result["source_documents"])
                else:
                    st.warning("Please enter a question.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.warning("Please upload a .txt document to begin.")