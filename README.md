# reading_agent
## Llama-Powered Text Analysis Agent

This project is a web-based agent built with Streamlit, Langchain, and a locally-run Llama model via Ollama. It allows users to upload a text document and perform two key tasks: generate a concise summary and ask specific questions about its content.
The core of this agent lies in its effective use of prompt and context engineering to ensure accurate, context-aware responses from the language model, even when running on a CPU without a dedicated GPU.

### Features


* 📄 Document Upload: Supports uploading .txt files for analysis.
* 📝 AI-Powered Summarization: Generates a concise summary of the entire document using a map-reduce strategy to handle large texts.
* ❓ Context-Aware Q&A: Ask questions about the document and get answers sourced directly from the text, minimizing factual inaccuracies (hallucinations).
* ⚙️ Configurable: Easily select different Llama models and adjust text processing parameters directly from the UI.
* 💻 Runs Locally: The entire stack, including the LLM, runs on your local machine, ensuring data privacy.

### Tech Stack
* LLM Runtime: Ollama for running Llama models (e.g., llama3:8b, mistral) locally.
* Application Framework: Streamlit for creating the interactive web UI.
* LLM Orchestration: Langchain for chaining LLM calls and managing the RAG pipeline.
* Vector Store: FAISS (Facebook AI Similarity Search) for fast, in-memory vector search (CPU-optimized).
* Embeddings: OllamaEmbeddings from langchain-community to generate text embeddings locally.

## Setup and Installation

Follow these steps to get the agent running on your local machine.

1. Prerequisites: Install Ollama
You must have Ollama installed and running.
Download from the official Ollama website.
After installation, pull a model. We recommend llama3:8b for a good balance of performance and capability on a CPU.
ollama pull llama3:8b
- Ensure the Ollama application is running in the background.

2. Clone the Repository & Install Dependencies
First, save the Python code as app.py in a new project directory. Then, install the required Python libraries.

#### Create a project directory
mkdir llama-text-agent && cd llama-text-agent

#### It's recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

#### Install the required packages
pip install streamlit langchain langchain-community faiss-cpu tiktoken

## How to Run

With Ollama running and the dependencies installed, launch the Streamlit application from your terminal:

    streamlit run app.py

Your web browser should automatically open a new tab with the application running.

## 🧠 How It Works: Prompt & Context Engineering
The agent's effectiveness comes from two specialized Langchain strategies designed to handle large texts and ensure factual grounding.

1. Question-Answering: Retrieval-Augmented Generation (RAG)
To answer questions accurately, the agent uses a RAG pipeline. This avoids stuffing the entire document into the model's limited context window.
* Chunking: The uploaded text is split into small, overlapping chunks.
* Embedding & Indexing: Each chunk is converted into a numerical vector (an embedding) using the local Llama model.  These vectors are stored in a FAISS vector store for efficient searching.
* Retrieval: When you ask a question, it's also converted into a vector. FAISS finds the text chunks with the most similar vectors, which are the parts of the text most relevant to your question.
* Prompting & Generation: These relevant chunks are inserted into a carefully engineered prompt that instructs the LLM to answer only based on the provided context. This significantly reduces the chance of the model making things up.

The Q&A prompt is designed to be robust:
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Context:
---
{context}
---

Question: {question}

Answer:

2. Summarization: Map-Reduce Chain
Summarizing a large document that doesn't fit in the model's context is handled with a map_reduce chain.
* Map Step: The agent "maps" over every text chunk and generates a short summary for each one individually.
* Reduce Step: It then takes all the individual summaries and "reduces" them by feeding them into the model one final time to create a single, coherent summary of the entire document.

## ⚙️ Configuration

* Select a Llama Model: Choose any model you have downloaded in Ollama.
* Text Chunk Size: Defines the size of each text chunk. Smaller chunks are processed faster but might lose some context between them.
* Text Chunk Overlap: Defines how many characters overlap between consecutive chunks. This helps maintain context and ensures that sentences aren't awkwardly split.