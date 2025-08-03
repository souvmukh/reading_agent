# core.py
import streamlit as st
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA, load_summarize_chain
from langchain.prompts import PromptTemplate

# --- 1. Model and Embeddings Initialization ---
@st.cache_resource
def initialize_components(model_name):
    """
    Initializes and caches the LLM and embeddings model to avoid reloading.
    """
    llm = Ollama(model=model_name, base_url="http://localhost:11434")
    embeddings = OllamaEmbeddings(model=model_name, base_url="http://localhost:11434")
    return llm, embeddings

# --- 2. Prompt handling ---

def get_summarize_prompt():
    """
    Returns the prompt template for the summarization task.
    """
    summarize_prompt_template = """
    Write a concise summary of the following text:
    ---
    {text}
    ---
    CONCISE SUMMARY:
    """
    return PromptTemplate(template=summarize_prompt_template, input_variables=["text"])

def get_qa_prompt():
    """
    Returns a robust prompt template for the Q&A task, engineered to prevent
    hallucination by grounding the model in the provided context.
    """
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
    return PromptTemplate(template=qa_prompt_template, input_variables=["context", "question"])

# --- 3. Core Text Processing and Chain Creation ---

def get_text_chunks(raw_text, chunk_size, chunk_overlap):
    """
    Splits the raw text into manageable chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(raw_text)

@st.cache_data
def create_vector_store(_texts, _embeddings):
    """
    Creates and caches a FAISS vector store from the text chunks.
    The `_` prefix in the arguments tells Streamlit to hash the *content* of the objects.
    """
    with st.spinner("Creating vector store for Q&A... This may take a moment."):
        vectorstore = FAISS.from_texts(texts=_texts, embedding=_embeddings)
    return vectorstore

def get_summary_chain(llm):
    """
    Builds and returns the summarization chain.
    """
    prompt = get_summarize_prompt()
    return load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=prompt,
        combine_prompt=prompt,
        verbose=False
    )

def get_qa_chain(llm, vectorstore):
    """
    Builds and returns the Retrieval-Augmented Generation (RAG) chain for Q&A.
    """
    prompt = get_qa_prompt()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
