# ui.py
import streamlit as st
import core  # Import the core logic module

def random_string(base_string, length=5):
    """
    Generates a random string of fixed length.
    This is a placeholder function; implement as needed.
    """
    import random
    import string
    return base_string + ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def setup_sidebar():
    """
    Sets up the Streamlit sidebar with configuration options.
    Returns the selected model and chunking parameters.
    """
    with st.sidebar:
        st.header("⚙️ Configuration")
        selected_model = st.selectbox(
            "Select a Llama Model",
            ["llama3:8b", "llama2", "phi3"],
            index=1,  # Default to "llama2"
            key=random_string("model-selection-", 5),
            help="Ensure the selected model is downloaded in Ollama."
        )
        chunk_size = st.slider(
            "Text Chunk Size", 500, 2000, 1000,
            key=random_string("chunk-size-", 5),
            help="Size of text chunks for processing."
        )
        chunk_overlap = st.slider(
            "Text Chunk Overlap", 0, 500, 200,
            key=random_string("chunk-overlap-", 5),
            help="Overlap between text chunks to maintain context."
        )
    return selected_model, chunk_size, chunk_overlap

def handle_summarization(llm, texts):
    """
    Handles the UI logic for the summarization tab.
    """
    st.subheader("Summarize the Document")
    if st.button("Generate Summary"):
        with st.spinner(f"Generating summary..."):
            # Create documents for the chain
            text_splitter = core.RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.create_documents(texts)
            
            # Get and run the chain
            summary_chain = core.get_summary_chain(llm)
            summary = summary_chain.run(docs)
            
            st.markdown("### Summary:")
            st.write(summary)

def handle_qa(llm, vectorstore):
    """
    Handles the UI logic for the Question-Answering tab.
    """
    st.subheader("Ask a Question Based on the Document")
    user_question = st.text_input("Enter your question here:")
    if st.button("Get Answer"):
        if user_question:
            with st.spinner(f"Searching for the answer..."):
                qa_chain = core.get_qa_chain(llm, vectorstore)
                result = qa_chain.invoke({"query": user_question})
                
                st.markdown("### Answer:")
                st.write(result["result"])

                with st.expander("Show source context"):
                    st.write(result["source_documents"])
        else:
            st.warning("Please enter a question.")

def main_page(llm, embeddings):
    """
    Renders the main page content, including file upload and tabs.
    """
    st.header("1. Upload Your Text Document")
    uploaded_file = st.file_uploader(
        "Upload a .txt file", type=["txt"],
        help="The agent will read, summarize, and answer questions based on this document."
    )

    if uploaded_file is not None:
        try:
            raw_text = uploaded_file.read().decode("utf-8")
            st.info("File successfully uploaded and read.")

            # Get configuration from sidebar
            _, chunk_size, chunk_overlap = setup_sidebar()
            
            # Process text and create vector store
            texts = core.get_text_chunks(raw_text, chunk_size, chunk_overlap)
            vectorstore = core.create_vector_store(texts, embeddings)
            st.success("Vector store created. Ready for summarization and Q&A.")

            # Display tabs
            tab1, tab2 = st.tabs(["📝 Summarize Document", "❓ Ask a Question"])
            with tab1:
                handle_summarization(llm, texts)
            with tab2:
                handle_qa(llm, vectorstore)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload a .txt document to begin.")
