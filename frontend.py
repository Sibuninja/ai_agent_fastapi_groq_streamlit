import streamlit as st
import requests

st.set_page_config(page_title="LangGraph Agent UI", layout="centered")
st.title("🤖 AI Chatbot Agents")
st.write("Create and interact with AI Agents powered by Groq or OpenAI!")

# --- Chatbot Section ---
system_prompt = st.text_area("Define your AI Agent:", height=70, placeholder="Type your system prompt here...")

MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
MODEL_NAMES_OPENAI = ["gpt-4o-mini"]

provider = st.radio("Select Provider:", ("Groq", "OpenAI"))

if provider == "Groq":
    selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
else:
    selected_model = st.selectbox("Select OpenAI Model:", MODEL_NAMES_OPENAI)

allow_web_search = st.checkbox("Allow Web Search")

user_query = st.text_area("Enter your query:", height=150, placeholder="Ask Anything!")

API_URL = "http://127.0.0.1:9999/chat"

if st.button("Ask Agent!"):
    if user_query.strip():
        payload = {
            "model_name": selected_model,
            "model_provider": provider,
            "system_prompt": system_prompt,
            "messages": [user_query],
            "allow_search": allow_web_search
        }

        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()

            try:
                response_data = response.json()
                if "error" in response_data:
                    st.error(f"❌ Error: {response_data['error']}")
                else:
                    final_answer = response_data.get("answer", response_data)
                    st.subheader("🧠 Agent Response")
                    st.markdown(f"**Final Answer:**\n\n{final_answer}")
            except Exception as json_error:
                st.error("❌ Failed to parse JSON response.")
                st.text("Raw response from server:")
                st.code(response.text)

        except requests.exceptions.RequestException as e:
            st.error(f"❌ API Request Failed: {e}")

# === PDF RAG Section ===
st.markdown("---")
st.header("📄 File-based RAG QA")

uploaded_file = st.file_uploader("Upload a PDF for Q&A", type="pdf")
if uploaded_file:
    with st.spinner("Uploading file..."):
        try:
            res = requests.post("http://127.0.0.1:9999/upload", files={"file": uploaded_file.getvalue()})
            if res.status_code == 200:
                st.success("✅ File uploaded successfully!")
            else:
                st.error("❌ Failed to upload file")
        except Exception as e:
            st.error(f"❌ Upload failed: {e}")

question = st.text_input("Ask a question about the uploaded PDF")
if st.button("Ask from PDF"):
    if question:
        with st.spinner("Processing..."):
            try:
                res = requests.post("http://127.0.0.1:9999/ask_rag", data={"question": question})
                if res.status_code == 200:
                    response = res.json()
                    st.markdown(f"**Answer:** {response.get('answer', 'No answer found')}")
                else:
                    st.error(f"❌ RAG API Error: {res.status_code}")
            except Exception as e:
                st.error(f"❌ RAG request failed: {e}")
