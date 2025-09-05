# backend.py
"""
FastAPI backend for LangGraph AI Agent (Groq/OpenAI).
- Validates incoming payloads (Pydantic).
- Runs blocking model + RAG work in background threads.
- Clear error messages for frontend to display.
- Upload endpoint saves PDF and prepares RAG vectors.
"""

import os
import asyncio
import logging
from typing import List, Optional, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Local modules (must exist in same project)
from ai_agent import get_response_from_ai_agent
from rag_engine import RAGPipeline
from utils_rag import query_groq

load_dotenv()  # load .env if present (do NOT commit .env to git)

logger = logging.getLogger("uvicorn.error")

# Allowed models on backend - keep in sync with frontend model list
ALLOWED_MODEL_NAMES = [
    "llama-3.3-70b-versatile",
    "gpt-4o-mini"
]

app = FastAPI(title="LangGraph AI Agent")

# CORS (open during dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Request models ===
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: Optional[str] = ""
    messages: List[ChatMessage]
    allow_search: bool = False

# RAG pipeline instance (created at startup)
rag: Optional[RAGPipeline] = None

@app.on_event("startup")
async def startup_event():
    """
    Initialize heavy components on startup (RAG pipeline).
    Running in a thread avoids blocking the event loop.
    """
    global rag
    logger.info("Starting up — initializing RAG pipeline (this may take a moment)...")
    # instantiate RAGPipeline in a thread
    try:
        rag = await asyncio.to_thread(RAGPipeline)
        logger.info("RAG pipeline initialized.")
    except Exception as e:
        logger.exception("Failed to initialize RAG pipeline at startup: %s", e)
        # keep going; we'll try on-demand during upload if needed

@app.get("/health")
def health():
    """Simple health endpoint."""
    return {"status": "ok"}

@app.post("/chat")
async def chat_endpoint(request: RequestState):
    """
    Chat endpoint expects a RequestState payload:
    - model_name: must be in ALLOWED_MODEL_NAMES
    - messages: list of ChatMessage objects (role/content)
    """
    # validate model selection
    if request.model_name not in ALLOWED_MODEL_NAMES:
        raise HTTPException(status_code=400, detail="Invalid model name. Kindly select a valid AI model.")

    # Normalize messages into simple dicts so ai_agent can accept strings/dicts/pydantic
    normalized_query = []
    for m in request.messages or []:
        normalized_query.append(m.dict())

    # Run the (blocking) model agent in a thread
    try:
        result = await asyncio.to_thread(
            get_response_from_ai_agent,
            request.model_name,
            normalized_query,
            request.allow_search,
            request.system_prompt or "",
            request.model_provider
        )

        # Ensure we always return a JSON object with "answer" key
        return {"answer": result}
    except ValueError as ve:
        # Input or configuration errors (bad model name, missing message, etc.)
        logger.warning("Client error in /chat: %s", ve)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Unexpected or downstream errors (model invocation, Groq errors, etc.)
        logger.exception("Unhandled exception in /chat")
        # Return a generic message to client while logging details server-side
        raise HTTPException(status_code=500, detail="Internal server error; check backend logs.")

# === RAG endpoints ===
UPLOAD_FOLDER = "./data/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Accepts PDF uploads and prepares them for RAG.
    Saves the file to UPLOAD_FOLDER and calls rag.prepare_doc(file_path)
    """
    global rag
    # Validate content-type if you like: file.content_type
    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        logger.info("📂 File saved: %s", file_path)

        # Ensure rag exists; if not, create it on demand in a thread
        if rag is None:
            logger.info("RAG not initialized yet — initializing on demand.")
            rag = await asyncio.to_thread(RAGPipeline)

        # Run prepare_doc in thread (may be CPU / I/O heavy)
        await asyncio.to_thread(rag.prepare_doc, file_path)
        return {"status": "uploaded", "filename": filename}
    except Exception as e:
        logger.exception("Upload failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

@app.post("/ask_rag")
async def ask_from_doc(question: str = Form(...)):
    """
    Query the prepared RAG index. This builds a prompt using retrieved chunks
    and calls query_groq(prompt) to get a model response.
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized yet.")

    logger.info("🔍 Received question for RAG: %s", question)
    try:
        # Build prompt (runs in thread to avoid blocking if retrieval is heavy)
        prompt = await asyncio.to_thread(rag.query_doc, question)
        logger.debug("Constructed prompt (short preview): %s", (prompt or "")[:300])

        # Call Groq (or other LLM wrapper) — this function should raise helpful exceptions
        result = await asyncio.to_thread(query_groq, prompt)
        logger.info("🤖 RAG model returned an answer.")
        return {"answer": result}
    except ValueError as ve:
        logger.warning("Client error in /ask_rag: %s", ve)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Error in /ask_rag")
        # If it's a Groq/API error, query_groq should throw something descriptive; return 502
        raise HTTPException(status_code=502, detail=str(e))

if __name__ == "__main__":
    # Run uvicorn programmatically so `python backend.py` starts the server.
    # Use reload=True for development only.
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=9999, reload=True)
