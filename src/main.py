import os
import sys
import time 
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# 1. GLOBAL CONFIGURATION & ENVIRONMENT
# ==========================================
# Load environment variables from .env file for secure API key management
load_dotenv()

# Define project structure paths relative to the script location
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "source_docs")

# Validate required credentials before execution
if not os.getenv("GOOGLE_API_KEY"):
    print("❌ Error: GOOGLE_API_KEY not found. Ensure your .env file is configured.")

def format_docs(docs):
    """
    Standardizes a list of retrieved documents into a single block of text
    for the LLM prompt context.
    """
    return "\n\n".join(doc.page_content for doc in docs)

# ==========================================
# 2. DATA INGESTION & VECTORIZATION
# ==========================================
def prepare_vector_store():
    """
    Handles the Extract-Transform-Load (ETL) pipeline:
    - Loads raw files from local storage
    - Splits text into manageable chunks
    - Generates local embeddings to avoid API rate limits
    - Stores vectors in a persistent ChromaDB
    """
    
    # Pre-flight check: Verify if supported files exist to prevent ChromaDB errors
    files = [f for f in os.listdir(DATA_PATH) if f.endswith(('.pdf', '.txt'))]
    if not files:
        print(f"⚠️  No valid documents found in {DATA_PATH}.")
        sys.exit()

    # Define specialized loaders for heterogeneous data types
    loaders = {
        ".pdf": DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader),
        ".txt": DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader),
    }
    
    # Load all documents into memory
    docs = []
    for loader in loaders.values():
        docs.extend(loader.load())
        
    # Split text into chunks to respect LLM context window limits
    # chunk_overlap ensures semantic continuity between chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
    # Utilize local embeddings for cost-efficiency and performance
    # 'all-MiniLM-L6-v2' is a lightweight, industry-standard transformer model
    local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    return Chroma.from_documents(
        documents=chunks,
        embedding=local_embeddings,
        persist_directory="./chroma_db",
    )

# ==========================================
# 3. AI ORCHESTRATION (LCEL PIPELINE)
# ==========================================
def get_rag_chain(vectorstore):
    """
    Constructs the Retrieval-Augmented Generation (RAG) chain using 
    LangChain Expression Language (LCEL).
    """
    
    # Initialize LLM with Temperature 0 for deterministic, logical reasoning
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
    
    # Configure retriever to fetch 'k' most relevant chunks for comprehensive context
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Reasoning-focused prompt to handle conflicting information and updates
    template = """You are a logical auditor. Your goal is to find the MOST ACCURATE and CURRENT answer 
    based ONLY on the documents provided.

    Follow these steps:
    1. List all mentions of a "password" found in the context.
    2. Check if any mention explicitly contradicts or updates a previous one.
    3. Identify the "real" or "final" version based on the text's logic.

    Context:
    {context}

    Question: {question}

    Final Answer (just the password):"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # LCEL 'Pipe' structure: Input -> Context Retrieval -> Prompt -> LLM -> Output
    # Passes retrieved docs into context directly, the chain contains a key 'context' (list of docs)
    # In RunnablePassthrough.assign, call format_docs to send formatted text to the prompt, but keep original docs in 'context'
    # Keeping original docs let API return source info later if needed
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | RunnablePassthrough.assign(
            answer=(
                (lambda x: {
                    "context": format_docs(x["context"]),
                    "question": x["question"]
                })
                | prompt
                | llm
                | StrOutputParser()
            )
        )
    )
    return chain

# ==========================================
# 4. EXECUTION ENTRY POINT
# ==========================================
if __name__ == "__main__":
    # Ensure local directory structure is initialized
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"✅ Initialized data directory: {DATA_PATH}")
    
    # Build the system components
    v_store = prepare_vector_store()
    rag_chain = get_rag_chain(v_store)
    
    # Interactive query loop
    print("\nAI Engine Ready. Ask a question about your documents:")
    user_query = input("> ")
    print("\nAI Response:", rag_chain.invoke(user_query))

    # Old chain

    '''
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    '''