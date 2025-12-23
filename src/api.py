from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel
from src.main import prepare_vector_store, get_rag_chain_with_history # Importing functions from main.py
import os

# Initialize FastAPI app
# 'app' object will be used to define API endpoints and connect web services
app = FastAPI(title="AI-Powered RAG System API")

# Initialize RAG components once server starts
# Do not reload vector store on every request (improves efficiency)
# Vector is now stored in RAM for quick access
print("--- Initializing AI Engine ---")
vector_store = prepare_vector_store()
rag_chain = get_rag_chain_with_history(vector_store)

# Define request model and expected type for incoming user question
# 'BaseModel' provides data validation and serialization, API auto reject bad data
class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default_user"

# Creates the API endpoint (URL path) to handle user questions
# When a POST request is sent to '/ask', this function is triggered
# POST is used when the client sends data to the server (the question)
# async def allows for asynchronous processing (non-blocking I/O). Multiple requests can be handled concurrently.
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Receives a JSON question, runs it through the RAG chain, 
    and returns the AI's response as JSON.
    """
    try:
        # Trigger the RAG logic from main.py
        response = rag_chain.invoke(
            {"question": request.question},
            config={"configurable": {"session_id": request.session_id}}
        )
        sources = []
        if "context" in response:
            for doc in response["context"]:
                source_path = doc.metadata.get("source", "Unknown")
                filename = os.path.basename(source_path)
                if filename not in sources:
                    sources.append(filename)

        # Return a structured JSON response
        return {
            "status": "success",
            "question": request.question,
            "answer": response,
            "session_id": request.session_id,
            "sources": sources
        }
    except Exception as e:
        # Handle errors gracefully and return error message
        if "429" in str(e):
            raise HTTPException(status_code=429, detail="AI is busy (Rate exceeded). Please try again in 60s.")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Health check endpoint to verify server is running
# GET is used to retrieve or "get" data from the server
@app.get("/")
def home():
    return {"message": "AI Knowledge Engine is Online"}

# Run this in your terminal to start the server with auto-reload enabled:
# uvicorn src.api:app --reload
# uvicorn is the ASGI server that runs FastAPI applications
# arc.api:app specifies the location of the FastAPI 'app' object from /src/api.py