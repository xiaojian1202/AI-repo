# AI-Powered RAG System with Stateful Memory

A robust Retrieval-Augmented Generation (RAG) engine designed to provide accurate, context-aware answers from local documents. This system features a "Logical Auditor" to handle conflicting data updates and maintains unique chat histories for multiple users via session-based memory.

## 🚀 Key Features

* **Logical Auditing:** Smart prompt engineering that allows the AI to prioritize the most recent or relevant information (e.g., tracking password updates over time).
* **Stateful Conversation:** Remembers previous questions and answers using session-managed memory, allowing for natural follow-up queries.
* **Local Vector Store:** Uses ChromaDB and HuggingFace embeddings to store and retrieve document context without sending private data to third-party embedding providers.
* **Source Citation:** Every answer includes the specific filenames used as context, ensuring transparency and reducing hallucinations.
* **RESTful API:** Powered by FastAPI with automated Swagger documentation for easy testing and integration.

## 🏗️ Architecture

The project follows a clean "Separation of Concerns" architecture:
1.  **Data Layer:** PDFs/Text files processed into chunks and stored in a persistent ChromaDB vector store.
2.  **Logic Layer (LangChain):** Orchestrates the retrieval of context, management of chat history, and interaction with the Gemini 2.0 Flash LLM.
3.  **Interface Layer (FastAPI):** Exposes the system as a web service with robust error handling for API rate limits.



## 🛠️ Tech Stack

* **Language:** Python 3.12
* **LLM:** Google Gemini 3.0 Flash Preview
* **Orchestration:** LangChain (LCEL)
* **Database:** ChromaDB (Vector Store)
* **API Framework:** FastAPI & Uvicorn
* **Embeddings:** HuggingFace (Sentence-Transformers)

## 🚦 Getting Started

### Prerequisites
* Python 3.12+
* A Google Gemini API Key

### Installation
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/ai-knowledge-engine.git](https://github.com/your-username/ai-knowledge-engine.git)
    cd ai-knowledge-engine
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up environment variables:**
    Create a `.env` file in the root and add your key:
    ```env
    GOOGLE_API_KEY=your_api_key_here
    ```

### Running the Server
Start the FastAPI server with auto-reload enabled:
```bash
uvicorn src.api:app --reload