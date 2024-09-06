# Document Question Answering System

This project implements a document question answering system using Llama and LangChain. It allows users to upload a document, ask questions about its content, and receive answers based on the document's information.

## Features

- Document loading and processing
- Text splitting for efficient processing
- Embedding generation using Llama
- Vector storage and similarity search using Chroma
- Question answering using a language model

## Prerequisites

- Python 3.7+
- Llama model (7B quantized version used in this example)
- LangChain library
- Streamlit (for the web interface)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/document-qa-system.git
   cd document-qa-system
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download the Llama model and place it in the `models` directory.

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Upload a document (currently supports .txt files)
3. Ask questions about the document content
4. View the AI-generated answers based on the document

## How it Works

1. The system loads the document and splits it into smaller chunks.
2. It generates embeddings for each chunk using the Llama model.
3. The embeddings are stored in a Chroma vector store for efficient retrieval.
4. When a question is asked, the system finds the most relevant document chunk using similarity search.
5. The relevant chunk and the question are passed to a language model to generate an answer.

## Customization

- You can adjust the chunk size and overlap in the `text_splitter` configuration.
- Different embedding models can be used by modifying the `embeddings` initialization.
- The prompt template can be customized to change how the model generates answers.
