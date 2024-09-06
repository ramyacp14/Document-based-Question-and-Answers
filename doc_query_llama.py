# Import required dependencies
import streamlit as st
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

MODEL_PATH = "./models/llama-7b.ggmlv3.q4_K_M.bin"

# Set up Streamlit app configuration
st.set_page_config(page_title="DocQ&A", page_icon="🤖", layout="wide")
st.markdown("""
            <style>
            .stApp {background-image: url("https://images.unsplash.com/photo-1468779036391-52341f60b55d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1968&q=80"); 
                    background-attachment: fixed; 
                    background-size: cover}
            </style>
            """, unsafe_allow_html=True)

# Function to save uploaded content as a temporary file
def save_temp_file(content, path):
    try:
        with open(path, 'w') as file:
            file.write(content)
        return True
    except Exception as error:
        print(f"Error saving file: {error}")
        return False

# Template for prompting the language model
prompt_template = """You are a helpful chatbot that answers questions using the given context. If the context does not provide enough information, say "Sorry, I don't know how to help with that."

Context sections:
{context}

Question:
{question}

Answer:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Initialize LLM and Embeddings
llm = LlamaCpp(model_path=MODEL_PATH)
embeddings = LlamaCppEmbeddings(model_path=MODEL_PATH)
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Application title and file uploader
st.title("📄 Document-Based Q&A")
st.text("Powered by Llama")
uploaded_file = st.file_uploader("Upload a text file", type="txt")
file_ready = False

if uploaded_file:
    # Read and save the file
    file_content = uploaded_file.read().decode('utf-8')
    temp_path = "temp/file.txt"
    save_temp_file(file_content, temp_path)
    
    # Load the document and process it
    loader = TextLoader(temp_path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=100)
    document_chunks = splitter.split_documents(docs)
    
    # Create vector database
    vector_db = Chroma.from_documents(document_chunks, embeddings)
    st.success("File loaded successfully!")
    file_ready = True

# Input query to retrieve relevant information
if file_ready:
    query = st.text_input("Ask a question about the document", placeholder="E.g., What does the document say about ...?", disabled=not uploaded_file)
    
    if query:
        # Search for relevant context
        search_results = vector_db.similarity_search(query, k=1)
        context = search_results[0].page_content
        
        # Run the query through the LLM and display the response
        response_chain = LLMChain(llm=llm, prompt=prompt)
        answer = response_chain.run({"context": context, "question": query})
        st.write(answer)
