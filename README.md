# Document Question Answering System

## Introduction

This project implements an advanced document question answering system using state-of-the-art language models and natural language processing techniques. By leveraging the power of Llama and LangChain, our system allows users to upload documents, ask questions about their content, and receive accurate, context-aware answers.

The system is designed to be efficient, scalable, and easily customizable, making it suitable for a wide range of applications, from personal knowledge management to enterprise-level document analysis.

## Features

- **Document Processing**: Support for various document formats (currently .txt, with plans to expand)
- **Intelligent Text Splitting**: Breaks down documents into manageable chunks while preserving context
- **Advanced Embedding Generation**: Utilizes Llama for creating high-quality text embeddings
- **Efficient Vector Storage**: Implements Chroma for fast similarity search and retrieval
- **Contextual Question Answering**: Employs a language model to generate accurate answers based on document context
- **User-Friendly Interface**: Built with Streamlit for easy interaction and visualization
- **Customizable Components**: Flexible architecture allowing for easy swapping of models and fine-tuning of parameters

## System Architecture

1. **Document Ingestion**: Documents are loaded using LangChain's `TextLoader`.
2. **Text Splitting**: The `RecursiveCharacterTextSplitter` breaks documents into smaller, overlapping chunks.
3. **Embedding Generation**: Llama generates embeddings for each text chunk.
4. **Vector Storage**: Chroma stores and indexes the embeddings for efficient retrieval.
5. **Query Processing**: User questions are embedded and compared against stored document embeddings.
6. **Context Retrieval**: The most relevant document chunks are retrieved using similarity search.
7. **Answer Generation**: A language model generates answers based on the retrieved context and user question.

## Prerequisites

- Python 3.7+
- Llama model (7B quantized version used in this example)
- CUDA-capable GPU (recommended for faster processing)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/document-qa-system.git
   cd document-qa-system
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Download the Llama model:
   - Visit [Hugging Face](https://huggingface.co/models) to download the appropriate Llama model
   - Place the model file in the `models` directory
   - Update the `MODEL_PATH` in the code to point to your model file

## Usage

1. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the provided local URL (typically `http://localhost:8501`)

3. Upload a document using the file uploader in the sidebar

4. Enter your question in the text input field

5. Click the "Ask" button to generate an answer

6. View the answer and relevant context in the main area of the app

## Configuration

Key configuration options can be found at the top of the `app.py` file:

- `MODEL_PATH`: Path to the Llama model file
- `CHUNK_SIZE`: Size of text chunks for splitting (default: 256)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 0)
- `TOP_K`: Number of most relevant chunks to consider (default: 1)

## Customization

- **Embedding Model**: Replace `LlamaCppEmbeddings` with other LangChain-compatible embedding models
- **Vector Store**: Swap Chroma with other vector stores like FAISS or Pinecone
- **Language Model**: Experiment with different LLMs supported by LangChain
- **Prompt Engineering**: Modify the `template` in the `PromptTemplate` to alter the system's response style

## Troubleshooting

- **Out of Memory Errors**: Reduce `CHUNK_SIZE` or use a smaller language model
- **Slow Performance**: Ensure you're using a GPU, or consider using a smaller/quantized model
- **Inaccurate Answers**: Experiment with different `CHUNK_SIZE` and `CHUNK_OVERLAP` values, or try a more advanced language model

## Contributing

We welcome contributions to improve the Document QA System! Here's how you can contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request
