from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from ai_agent import get_response_from_ai_agent
from rag_engine import RAGPipeline
from utils_rag import query_groq
import os
from dotenv import load_dotenv

load_dotenv()

# === AGENT CONFIG ===
ALLOWED_MODEL_NAMES = [
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
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

# === AI Chat Endpoint ===
class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

@app.post("/chat")
def chat_endpoint(request: RequestState):
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": "Invalid model name. Kindly select a valid AI model."}

    try:
        response = get_response_from_ai_agent(
            llm_id=request.model_name,
            query=request.messages,
            allow_search=request.allow_search,
            system_prompt=request.system_prompt,
            provider=request.model_provider
        )

        # ✅ Always return a dict (to match frontend expectation)
        return {"answer": response}
    
    except Exception as e:
        return {"error": str(e)}

# === RAG Setup ===
UPLOAD_FOLDER = "./data/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
rag = RAGPipeline()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    print("📂 File saved:", file_path)
    rag.prepare_doc(file_path)
    return {"status": "uploaded", "filename": file.filename}

@app.post("/ask_rag")
async def ask_from_doc(question: str = Form(...)):
    print("🔍 Received question:", question)
    prompt = rag.query_doc(question)
    print("📄 Constructed prompt:\n", prompt)

    try:
        answer = query_groq(prompt)
        print("🤖 Groq answer:", answer)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
