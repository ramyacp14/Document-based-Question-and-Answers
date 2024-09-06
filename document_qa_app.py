import os
import streamlit as st
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Hugging Face API token
huggingfacehub_api_token = os.getenv("HUGGINGFACE_API_TOKEN")

if huggingfacehub_api_token is None:
    st.error("Hugging Face API token is not set. Please check your environment variables.")

# Customize the layout
st.set_page_config(page_title="DocQA", page_icon="🤖", layout="wide")
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1468779036391-52341f60b55d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1968&q=80");
        background-attachment: fixed;
        background-size: cover
    }
    </style>
    """, unsafe_allow_html=True)

# Ensure temp directory exists
if not os.path.exists("temp"):
    os.makedirs("temp")

# Function to write uploaded file
@st.cache_data
def write_text_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)
    return True

# Function to check if file is docx
def is_docx_file(file):
    return file.name.lower().endswith('.docx')

# Set prompt template
prompt_template = """
You are an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. Below is some information. 
{context}

Based on the above information only, answer the below question. 

Question: {question}
Answer:"""

prompt = PromptTemplate.from_template(prompt_template)

# Streamlit app
st.title("📄 Document Question Answering")
st.text("𓅃 Powered by Hugging Face Models")

# Model selection
model_options = {
    'Falcon-7B': "tiiuae/falcon-7b-instruct",
    'Dolly-v2-3B': "databricks/dolly-v2-3B",
    'FLAN-T5-XL': "google/flan-t5-xl"
}

selected_model = st.selectbox('Which model would you like to use?', list(model_options.keys()))
repo_id = model_options[selected_model]

# Initialize language model
@st.cache_resource
def get_llm(repo_id):
    return HuggingFaceHub(
        huggingfacehub_api_token=huggingfacehub_api_token,
        repo_id=repo_id,
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

llm = get_llm(repo_id)

# Initialize embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings()

embeddings = get_embeddings()

# File uploader
uploaded_file = st.file_uploader("Upload an article", type=["txt", "docx"])

if uploaded_file is not None:
    if is_docx_file(uploaded_file):
        # Add docx handling logic here
        st.error("DOCX support not implemented yet.")
    else:
        content = uploaded_file.read().decode('utf-8')
        file_path = os.path.join("temp", uploaded_file.name)
        write_text_file(content, file_path)
        
        # Load and process the document
        loader = TextLoader(file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.split_documents(docs)
        
        # Create vector store
        db = FAISS.from_documents(texts, embeddings)
        st.success("File Loaded Successfully!")
        
        # Query through LLM
        question = st.text_input("Ask a question about the document:", disabled=not uploaded_file)
        
        if question:
            similar_docs = db.similarity_search(question, k=2)
            context = "\n".join([doc.page_content for doc in similar_docs])
            
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            response = llm_chain.run(context=context, question=question)
            
            st.write("Answer:", response)
            
            with st.expander("View relevant context"):
                st.write(context)

# Add a footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and LangChain")
