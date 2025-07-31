# Snapp Driver Assistant Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions from Snapp (the largest ride-hailing app in Iran) drivers. The chatbot uses a knowledge base created by scraping the official Snapp driver training and documentation website.

The primary goal is to provide accurate, context-aware answers to driver-specific queries based solely on the official documentation, preventing the model from hallucinating or providing irrelevant information.

## How It Works

The project is built on a three-stage pipeline:

**1. Data Scraping (`scrapper.ipynb`)**
- A Jupyter Notebook using **Selenium** and **BeautifulSoup** crawls the [Snapp Driver Club](https://club.snapp.ir/training-center/) website.
- It navigates through different categories and sub-categories of help articles.
- The script extracts the text content from each page and stores it in a structured JSON file named `docs.json`.

**2. Embedding Generation (`embd.py`)**
- This script processes the raw text from `docs.json`.
- It uses the `hazm` library for Persian text preprocessing, including normalization, tokenization, and stopword removal.
- The cleaned text for each document is then converted into a high-dimensional vector (embedding) using the `paraphrase-multilingual-mpnet-base-v2` model from **SentenceTransformers**.
- The script outputs two files:
    - `embeddings.json`: A dictionary mapping each document title to its vector embedding.
    - `full_doc.json`: A dictionary mapping each document title to its full, cleaned text, ready for retrieval.

**3. Chat Application (`app.py`)**
- This is the main RAG application powered by **LangChain**.
- When a user asks a question:
    - The user's query is preprocessed and converted into an embedding using the same sentence transformer model.
    - **Scikit-learn's `cosine_similarity`** is used to compare the query's embedding against all document embeddings in `embeddings.json`.
    - The top 3 most relevant documents are retrieved from `full_doc.json`.
    - The retrieved documents are injected as context into a prompt template.
    - An LLM (in this case, accessed via the `ChatOpenAI` wrapper) generates a final answer based *only* on the provided context.
- This ensures the answers are grounded in the scraped documentation.

## Technology Stack

- **Web Scraping:** Selenium, BeautifulSoup4
- **Data Handling:** NumPy, Pandas
- **NLP & Embeddings:** SentenceTransformers, Hazm
- **Machine Learning:** Scikit-learn
- **LLM Framework:** LangChain (with `ChatOpenAI`)
