from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import shutil
import os
# Note: Ensure you are running this from the root folder 
# (uvicorn backend.main:app) so these imports work.
from rag_engine import RAGManager
from database import init_db, add_message, get_chat_history

app = FastAPI(title="RAG Agent Backend")

# Initialize Systems
rag_manager = RAGManager()
init_db()

# Create data directory to store uploaded files temporarily
os.makedirs("data", exist_ok=True)

# Update Query Request Model to include Username (Identity)
class QueryRequest(BaseModel):
    query: str
    session_id: str
    username: str   # NEW: Who is asking?
    provider: str   # 'openai' or 'gemini'
    api_key: str

@app.post("/upload/")
async def upload_files(
    files: List[UploadFile] = File(...),
    username: str = Form(...),  # NEW: Owner of the file
    privacy: str = Form(...),   # NEW: 'private' or 'public'
    provider: str = Form(...),
    api_key: str = Form(...)
):
    """
    Endpoint to upload PDF/TXT files, tag them with User/Privacy, 
    and ingest them into the Persistent Vector DB.
    """
    saved_paths = []
    try:
        # Save files to disk temporarily
        for file in files:
            file_location = f"data/{file.filename}"
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(file_location)
        
        # Process files with Metadata (Username & Privacy)
        # Note: We pass provider/api_key to init embeddings if needed
        num_chunks = rag_manager.process_files(
            file_paths=saved_paths,
            username=username,
            privacy=privacy,
            provider=provider,
            api_key=api_key
        )
        
        return {
            "status": "success", 
            "message": f"Successfully processed {len(files)} files into {num_chunks} chunks for user '{username}' ({privacy} mode)."
        }
    
    except Exception as e:
        # Log error to console for debugging
        print(f"Error in upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
async def chat_endpoint(request: QueryRequest):
    """
    Endpoint to handle queries with history, RAG context, and Privacy Filtering.
    """
    try:
        # 1. Get Chat History from SQLite
        history = get_chat_history(request.session_id)
        
        # 2. Get Answer from RAG Engine with Privacy Check
        result = rag_manager.get_answer(
            query=request.query,
            history=history,
            username=request.username, # Pass username to filter private docs
            provider=request.provider,
            api_key=request.api_key
        )
        
        # 3. Save Conversation to Memory (SQLite)
        add_message(request.session_id, "user", request.query)
        add_message(request.session_id, "assistant", result["answer"])
        
        return result
        
    except Exception as e:
        print(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))