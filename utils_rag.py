# utils_rag.py
import fitz  # PyMuPDF; ensure `pip install pymupdf`
import requests
import os
import json
import time

# Configurable RAG model via env (override in .env)
DEFAULT_GROQ_RAG_MODEL = os.environ.get("GROQ_RAG_MODEL", "gpt-4o-mini")
# A short ordered fallback list (tweak to the models your account supports)
FALLBACK_GROQ_MODELS = [DEFAULT_GROQ_RAG_MODEL, "llama-3.3-70b-versatile"]

# ---------- PDF helpers ----------
def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file using PyMuPDF.
    """
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text, max_length=500):
    """
    Splits a large text string into chunks of approx max_length words.
    """
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ---------- Groq query ----------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    # do not raise here; raise when function is used, so imports don't crash
    pass


def _call_groq_chat(messages, model):
    """
    Low-level call to Groq chat completions endpoint.
    Returns (status_code, parsed_json_or_text)
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages, "temperature": 0.0}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
    except Exception as e:
        return None, {"error": f"Request failed: {e}"}
    try:
        return resp.status_code, resp.json()
    except Exception:
        return resp.status_code, resp.text


def query_groq(prompt: str) -> str:
    """
    Sends the prompt to Groq chat model. Tries DEFAULT_GROQ_RAG_MODEL then fallbacks.
    Raises Exception with useful messages on failure.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Set it in your .env or environment.")

    messages = [
        {"role": "system", "content": "You are a helpful assistant answering from the provided context."},
        {"role": "user", "content": prompt}
    ]

    tried = []
    candidates = FALLBACK_GROQ_MODELS.copy()
    # ensure unique while preserving order
    seen = set()
    candidates = [m for m in candidates if not (m in seen or seen.add(m))]

    for model in candidates:
        tried.append(model)
        status, resp = _call_groq_chat(messages, model)
        if status == 200:
            # handle safe parsing
            try:
                content = resp.get("choices", [{}])[0].get("message", {}).get("content")
                if content:
                    return content.strip()
            except Exception:
                return json.dumps(resp)
        # model-specific errors -> try next candidate
        if status in (400, 404) or (isinstance(resp, dict) and resp.get("error")):
            err_msg = ""
            try:
                err_msg = resp.get("error", {}).get("message", str(resp))
            except Exception:
                err_msg = str(resp)
            lowered = err_msg.lower()
            if "decommissioned" in lowered or "model_not_found" in lowered or "does not exist" in lowered or "not found" in lowered:
                print(f"⚠️ Groq: model '{model}' not available ({err_msg}). Trying next fallback. Tried: {tried}")
                time.sleep(0.5)
                continue
            # other client errors: raise immediately with details
            raise Exception(f"Groq API error for model '{model}': {status} - {err_msg}")

        # other unknown HTTP status -> raise
        raise Exception(f"Groq API unexpected response for model '{model}': status={status}, resp={resp}")

    raise Exception(f"Groq API failed for all tried models: {tried}. Check your GROQ_API_KEY and model access.")
