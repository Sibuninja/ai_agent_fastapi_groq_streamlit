# utils_rag.py

import fitz  # PyMuPDF
import requests
import os

def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file using PyMuPDF.
    Returns the full text as a single string.
    """
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text, max_length=500):
    """
    Splits a large text string into smaller chunks of words (default 500).
    Useful for embedding and retrieval in RAG pipelines.
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


import os
import requests

def query_groq(prompt, model="llama3-70b-8192"):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("❌ GROQ_API_KEY is missing. Please check your .env file.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)

    if response.status_code != 200:
        print("❌ Groq API Error:", response.status_code)
        print("🔍 Response text:", response.text)
        raise Exception("Groq API request failed.")

    return response.json()["choices"][0]["message"]["content"].strip()


    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content'].strip()
