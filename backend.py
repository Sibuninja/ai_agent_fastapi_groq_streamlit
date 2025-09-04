# backend.py
# FastAPI backend with robust normalization and friendly error returns.

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Literal
from ai_agent import get_response_from_ai_agent
from rag_engine import RAGPipeline
from utils_rag import query_groq
import os
from dotenv import load_dotenv
import asyncio
import uvicorn
import logging

load_dotenv()

logger = logging.getLogger("uvicorn.error")

# === AGENT CONFIG ===
ALLOWED_MODEL_NAMES = [
    "llama-3.3-70b-versatile",
    "gpt-4o-mini"
]

app = FastAPI(title="LangGraph AI Agent")

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === AI Chat Endpoint request models ===
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[ChatMessage]
    allow_search: bool

# RAG instance created on startup
rag: Optional[RAGPipeline] = None

@app.on_event("startup")
async def startup_event():
    global rag
    logger.info("Starting up — initializing RAG pipeline (this may take a moment)...")
    # instantiate RAGPipeline in a thread so startup doesn't block the event loop too long
    rag = await asyncio.to_thread(RAGPipeline)
    logger.info("RAG pipeline initialized.")

@app.post("/chat")
async def chat_endpoint(request: RequestState):
    if request.model_name not in ALLOWED_MODEL_NAMES:
        raise HTTPException(status_code=400, detail="Invalid model name. Kindly select a valid AI model.")

    # Normalize Pydantic ChatMessage -> dict or keep strings if any
    normalized_query = []
    for m in request.messages or []:
        if hasattr(m, "dict"):
            normalized_query.append(m.dict())
        else:
            normalized_query.append(m)

    try:
        # run blocking agent in a thread
        response = await asyncio.to_thread(
            get_response_from_ai_agent,
            request.model_name,
            normalized_query,
            request.allow_search,
            request.system_prompt,
            request.model_provider
        )
        return {"answer": response}
    except ValueError as ve:
        # client/input-side error (missing key, model access, etc.)
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        # model invocation or downstream failure
        raise HTTPException(status_code=502, detail=str(re))
    except Exception as e:
        logger.exception("Unhandled exception in /chat")
        raise HTTPException(status_code=500, detail="Internal server error")

# === RAG Setup ===
UPLOAD_FOLDER = "./data/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        logger.info("📂 File saved: %s", file_path)

        if rag is None:
            logger.info("RAG not initialized yet — initializing on demand.")
            await asyncio.to_thread(lambda: globals().__setitem__('rag', RAGPipeline()))

        await asyncio.to_thread(rag.prepare_doc, file_path)
        return {"status": "uploaded", "filename": file.filename}
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask_rag")
async def ask_from_doc(question: str = Form(...)):
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized yet.")

    logger.info("🔍 Received question: %s", question)
    prompt = await asyncio.to_thread(rag.query_doc, question)
    logger.debug("📄 Constructed prompt: %s", prompt)

    try:
        answer = await asyncio.to_thread(query_groq, prompt)
        logger.info("🤖 Groq answer received.")
        return {"answer": answer}
    except Exception as e:
        logger.exception("Error in /ask_rag")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("backend:app", host="127.0.0.1", port=9999, reload=True)
