# frontend.py
# Streamlit UI for AI Chatbot Agents with minimal changes for Render deployment

import os
import streamlit as st
import requests

# --- Page Config ---
st.set_page_config(page_title="LangGraph Agent UI", layout="centered")
st.title("🤖 AI Chatbot Agents")
st.write("Create and interact with AI Agents powered by Groq or OpenAI!")

# --- Chatbot Section ---
system_prompt = st.text_area(
    "Define your AI Agent:", height=70, placeholder="Type your system prompt here..."
)

MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile"]
MODEL_NAMES_OPENAI = ["gpt-4o-mini"]

provider = st.radio("Select Provider:", ("Groq", "OpenAI"))

selected_model = (
    st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
    if provider == "Groq"
    else st.selectbox("Select OpenAI Model:", MODEL_NAMES_OPENAI)
)

allow_web_search = st.checkbox("Allow Web Search")
user_query = st.text_area("Enter your query:", height=150, placeholder="Ask Anything!")

# --- Backend URLs (Use Environment Variable or Default to Local) ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:9999")
API_URL = f"{BACKEND_URL}/chat"
upload_url = f"{BACKEND_URL}/upload"
ask_rag_url = f"{BACKEND_URL}/ask_rag"

# --- Chatbot Request ---
if st.button("Ask Agent!"):
    if not user_query.strip():
        st.warning("Please enter a query before asking the agent.")
    else:
        payload = {
            "model_name": selected_model,
            "model_provider": provider,
            "system_prompt": system_prompt,
            "messages": [{"role": "user", "content": user_query}],
            "allow_search": allow_web_search,
        }

        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            response.raise_for_status()
            response_data = response.json()

            if isinstance(response_data, dict) and "detail" in response_data:
                st.error(f"❌ Error: {response_data['detail']}")
            elif isinstance(response_data, dict) and "error" in response_data:
                st.error(f"❌ Error: {response_data['error']}")
            else:
                final_answer = response_data.get("answer", response_data)
                st.subheader("🧠 Agent Response")
                st.markdown(f"**Final Answer:**\n\n{final_answer}")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ API Request Failed: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

# --- PDF RAG Section ---
st.markdown("---")
st.header("📄 File-based RAG QA")

uploaded_file = st.file_uploader("Upload a PDF for Q&A", type="pdf")
if uploaded_file:
    with st.spinner("Uploading file..."):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            res = requests.post(upload_url, files=files, timeout=60)
            if res.status_code == 200:
                st.success("✅ File uploaded successfully!")
            else:
                st.error(f"❌ Failed to upload file (status {res.status_code}): {res.text}")
        except Exception as e:
            st.error(f"❌ Upload failed: {e}")

question = st.text_input("Ask a question about the uploaded PDF")
if st.button("Ask from PDF"):
    if not question:
        st.warning("Please enter a question about the uploaded PDF.")
    else:
        with st.spinner("Processing..."):
            try:
                res = requests.post(ask_rag_url, data={"question": question}, timeout=60)
                if res.status_code == 200:
                    response = res.json()
                    st.markdown(f"**Answer:** {response.get('answer', 'No answer found')}")
                else:
                    st.error(f"❌ RAG API Error: {res.status_code} - {res.text}")
            except Exception as e:
                st.error(f"❌ RAG request failed: {e}")
