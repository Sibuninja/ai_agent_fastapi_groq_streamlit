# AI Agent FastAPI Groq Streamlit

A powerful AI chatbot application built with FastAPI, Groq LLM, and Streamlit that provides intelligent conversation capabilities with optional web search functionality.

## Features

- 🤖 **AI Chatbot**: Powered by Groq's fast LLM models
- 🔍 **Web Search**: Integration with Tavily for real-time information
- ⚡ **FastAPI Backend**: High-performance API server
- 🎨 **Streamlit Frontend**: Interactive web interface
- 🔧 **Multi-Provider Support**: Support for both Groq and OpenAI models
- 🌐 **RESTful API**: Easy integration with other applications

## Tech Stack

- **Backend**: FastAPI + Python
- **Frontend**: Streamlit
- **AI Models**: Groq (LLaMA, Mixtral) / OpenAI (GPT)
- **Search**: Tavily Search API
- **Framework**: LangChain + LangGraph

## Prerequisites

- Python 3.8 or higher
- Git
- API Keys (see setup section)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ai_agent_fastapi_groq_streamlit.git
cd ai_agent_fastapi_groq_streamlit
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables

Create a `.env` file in the project root:

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional (for search functionality)
TAVILY_API_KEY=your_tavily_api_key_here

# Optional (if using OpenAI models)
OPENAI_API_KEY=your_openai_api_key_here
```

## API Keys Setup

### Groq API Key (Required)
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy and paste into your `.env` file

### Tavily API Key (Optional - for search)
1. Visit [tavily.com](https://tavily.com)
2. Sign up for an account
3. Get your API key from the dashboard
4. Add to your `.env` file

### OpenAI API Key (Optional - alternative to Groq)
1. Visit [platform.openai.com](https://platform.openai.com)
2. Create an account and get API key
3. Add to your `.env` file

## Running the Application

You need to run **three components** in separate terminals:

### Terminal 1: AI Agent
```bash
python ai_agent.py
```

### Terminal 2: Frontend (Streamlit)
```bash
streamlit run frontend.py
```

### Terminal 3: Backend (FastAPI)
```bash
python backend.py
```

## Accessing the Application

Once all components are running:

- **Streamlit Web Interface**: http://localhost:8501
- **FastAPI Backend**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Available Models

### Groq Models
- `llama-3.3-70b-versatile` (Default)
- `llama3-8b-8192`
- `llama3-70b-8192`
- `mixtral-8x7b-32768`
- `gemma-7b-it`

### OpenAI Models (if configured)
- `gpt-4o-mini`
- `gpt-4o`
- `gpt-3.5-turbo`

## Project Structure

```
ai_agent_fastapi_groq_streamlit/
├── ai_agent.py          # Core AI agent logic
├── backend.py           # FastAPI server
├── frontend.py          # Streamlit interface
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create this)
├── README.md           # This file
└── venv/               # Virtual environment
```

## Usage

1. Start all three components as described above
2. Open your browser to http://localhost:8501
3. Type your message in the chat interface
4. Toggle search functionality on/off as needed
5. Select different AI models from the dropdown

## API Endpoints

### FastAPI Endpoints
- `GET /`: Health check
- `POST /chat`: Send message to AI agent
- `GET /docs`: Interactive API documentation

## Troubleshooting

### Common Issues

**1. Module not found errors:**
```bash
pip install fastapi uvicorn streamlit groq python-dotenv langchain-groq langchain-community langgraph tavily-python
```

**2. Port already in use:**
```bash
# Run backend on different port
uvicorn backend:app --port 8001

# Run frontend on different port
streamlit run frontend.py --server.port 8502
```

**3. API key errors:**
- Ensure your `.env` file is in the project root
- Check that API keys are valid and have sufficient credits
- Verify the `.env` file format (no spaces around `=`)

**4. Streamlit script context warnings:**
- Always use `streamlit run frontend.py` instead of `python frontend.py`

### Testing Installation

```bash
# Test if AI agent loads
python -c "import ai_agent; print('AI Agent loaded successfully!')"

# Test environment variables
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('GROQ_API_KEY:', 'Found' if os.getenv('GROQ_API_KEY') else 'Missing')"
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

If you encounter any issues:

1. Check the troubleshooting section above
2. Ensure all API keys are correctly configured
3. Verify all three components are running
4. Check the console outputs for specific error messages

## Acknowledgments

- [Groq](https://groq.com/) for fast LLM inference
- [LangChain](https://langchain.com/) for AI framework
- [Streamlit](https://streamlit.io/) for the web interface
- [FastAPI](https://fastapi.tiangolo.com/) for the backend API
- [Tavily](https://tavily.com/) for search capabilities

---

**Happy Chatting! 🚀**