# Web Retrieval Q&A System

This project allows users to input a website URL, processes the content of that website, and then answers questions asked by the user based *only* on the retrieved content from that specific URL. It utilizes a Retrieval-Augmented Generation (RAG) approach.

## Overview

The script performs the following steps:

1.  **Fetches Web Content:** Takes a URL as input and scrapes the textual content from the corresponding webpage using `requests` and `BeautifulSoup`.
2.  **Text Chunking:** Splits the extracted text into smaller, manageable chunks using `RecursiveCharacterTextSplitter` from Langchain.
3.  **Embedding Generation:** Creates vector embeddings for each text chunk using a sentence transformer model (`sentence-transformers/all-MiniLM-L6-v2`) via `HuggingFaceEmbeddings`.
4.  **Vector Store Creation:** Stores the text chunks and their corresponding embeddings in a FAISS vector store for efficient similarity searching.
5.  **Retrieval:** When a user asks a question, the system searches the FAISS index to find the text chunks most relevant to the query.
6.  **Answer Generation:** Uses a pre-trained causal language model (`Qwen/Qwen2.5-0.5B-Instruct` from Hugging Face Transformers) to generate an answer. The model is prompted with the user's query and the relevant text chunks retrieved in the previous step as context.
7.  **Interactive Q&A:** Provides a simple command-line interface for users to enter a URL and then ask multiple questions about its content.

## Features

* Scrapes text content from any given public URL.
* Builds an in-memory vector database (FAISS) for the scraped content.
* Retrieves relevant context based on user queries.
* Generates answers using a powerful language model constrained by the retrieved context.
* Handles basic errors during web fetching and processing.
* Includes rudimentary handling for potential CUDA memory issues.

## Requirements

* Python 3.8+
* Libraries listed in `requirements.txt`

## Installation

1.  **Clone the repository (Optional):**
    ```bash
    git clone https://github.com/NavK01/Web_Retrieval.git
    cd Web_Retrieval
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  
    ```

    # On Windows use 
    ```bash
    python -m venv venv

    venv\Scripts\activate 
    ```

3.  **Install Dependencies:**
    

    Install the packages:
    ```bash
    pip install -r requirements.txt
    ```

    **Note on FAISS:**
    * `faiss-cpu` is for CPU-only usage.
    * `faiss-gpu` requires an NVIDIA GPU and a proper CUDA environment setup. If you have issues, start with `faiss-cpu`.

    **Note on Language Model:**
    * The script uses `Qwen/Qwen2.5-0.5B-Instruct`. This model will be downloaded automatically on first use. Ensure you have sufficient disk space and potentially RAM/VRAM.
    * Using `device_map="auto"` requires `accelerate` and PyTorch. It will attempt to use a GPU if available and configured correctly; otherwise, it will fall back to the CPU, which might be very slow for inference.

## Usage

1.  **Run the script:**
    ```bash
    python Web_Retrieval.py # Replace your_script_name.py with the actual filename
    ```

2.  **Enter URL:**
    When prompted, enter the full URL of the website you want to query (e.g., `https://en.wikipedia.org/wiki/Artificial_intelligence`). The script will then fetch, process, and index the content. This might take a moment depending on the size of the webpage and your machine's speed.

3.  **Ask Questions:**
    Once the vector database is ready, you'll be prompted to enter your questions. Type your question and press Enter.

4.  **Get Answers:**
    The script will retrieve relevant context from the webpage content and generate an answer based on that context. The question and the generated answer will be displayed.

5.  **Exit:**
    To exit the Q&A loop, enter `1` when prompted for a question.

## How It Works (RAG Pipeline)

1.  **Load:** Fetch data from the specified URL.
2.  **Split:** Break the loaded document into smaller chunks.
3.  **Store:** Embed each chunk and load them into a FAISS vector store.
4.  **Retrieve:** Given a user query, retrieve the relevant chunks from the store.
5.  **Generate:** Pass the query and the retrieved chunks to the language model (`Qwen/Qwen2.5-0.5B-Instruct`) to generate an answer based *only* on the provided context.

## Configuration Options (Inside the script)

* **`RecursiveCharacterTextSplitter` parameters:** `chunk_size` and `chunk_overlap` can be adjusted to change how the text is divided.
* **Embedding Model:** `HuggingFaceEmbeddings(model_name=...)` can be changed to use a different sentence transformer model.
* **Vector Store:** Currently uses FAISS. Other Langchain-compatible vector stores could be swapped in.
* **Retrieval Parameter (`k`):** The `get_relevant_text` function retrieves the top `k` (default is 2) most relevant chunks. This can be adjusted.
* **Language Model:** The `AutoTokenizer` and `AutoModelForCausalLM` can be pointed to different Hugging Face models. Ensure the chosen model is suitable for instruction following or question answering. Model size will impact performance and resource requirements (RAM/VRAM).
* **Prompt:** The prompt template within `answer_query_with_context` guides the LLM. It can be modified for different response styles or constraints.

