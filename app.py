import os
import streamlit as st 
from docx import Document
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Set up the HuggingFace API token from environment variables
huggingfacehub_api_token = os.getenv("HUGGINGFACE_EDU")

# Set Streamlit app configuration
st.set_page_config(page_title="Document Q&A", page_icon="🤖", layout="wide")     
st.markdown("""
            <style>
            .stApp {background-image: url("https://images.unsplash.com/photo-1468779036391-52341f60b55d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1968&q=80"); 
                     background-attachment: fixed;
                     background-size: cover}
            </style>
            """, unsafe_allow_html=True)

# Function to save the uploaded file to a temp directory
def save_file(content, path):
    try:
        with open(path, 'w') as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False

# Function to check if the uploaded file is a .docx format
def is_docx(file):
    return file.name.split(".")[-1].lower() == "docx"

# Define prompt template for the LLM
template = """
You are a helpful and polite AI assistant. Below is some context information. 
{context}

Based on the provided information, please answer the following question: 

{question}
"""
prompt = PromptTemplate.from_template(template)

# Display the app title and select box for model selection
st.title("📄 Document-Based Q&A")
st.text("𓅃 Powered by Falcon-7B")

model_choice = st.selectbox(
    'Choose the language model:',
    ('Falcon-7B', 'Dolly-v2-3B'))

model_repo = {'Falcon-7B': "tiiuae/falcon-7b-instruct", 'Dolly-v2-3B': "databricks/dolly-v2-3b"}
chosen_model = model_repo[model_choice]

# Set up HuggingFaceHub model
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token, 
                     repo_id=chosen_model, 
                     model_kwargs={"temperature": 0.6, "max_new_tokens": 250 if model_choice == 'Dolly-v2-3B' else 500})

embeddings = HuggingFaceEmbeddings()
llm_chain = LLMChain(prompt=prompt, llm=llm)

# File uploader widget
uploaded_file = st.file_uploader("Upload a document (.docx or .txt)", type=["docx", "txt"])
file_ready = False

# File processing and loading
if uploaded_file:
    if is_docx(uploaded_file):
        doc = Document(uploaded_file)
        file_text = "\n".join([p.text for p in doc.paragraphs])
        file_path = "temp/document.txt"
        save_file(file_text, file_path)
    else:
        file_text = uploaded_file.read().decode('utf-8')
        file_path = "temp/document.txt"
        save_file(file_text, file_path)

    # Load the document and split it into chunks
    loader = TextLoader(file_path)
    documents = loader.load()    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256, chunk_overlap=0, separators=[" ", ",", "\n", "."]
    )
    chunks = text_splitter.split_documents(documents)
    vector_db = Chroma.from_documents(chunks, embeddings)    
    st.success("File loaded successfully!")
    file_ready = True

# Process user queries when the file is ready
if file_ready:
    query = st.text_input("Ask a question based on the document", placeholder="Example: Find references to ...", disabled=not uploaded_file)
    if query:
        # Perform similarity search
        results = vector_db.similarity_search(query, k=1)
        context = results[0].page_content

        # Run the query through the language model
        response_chain = LLMChain(llm=llm, prompt=prompt)
        response = response_chain.run({"context": context, "question": query})

        # Display the result
        st.write(response)
