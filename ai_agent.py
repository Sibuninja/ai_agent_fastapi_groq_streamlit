# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Step1: Setup API Keys for Groq, OpenAI and Tavily
import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Step2: Setup LLM & Tools
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Initialize Groq LLM
groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY,
    temperature=0.1
)

# Initialize search tool
search_tool = TavilySearchResults(max_results=2)

# Step3: Setup AI Agent with Search tool functionality
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

system_prompt = "Act as an AI chatbot who is smart and friendly"

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider == "Groq":
        llm = ChatGroq(model=llm_id, groq_api_key=GROQ_API_KEY)
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=llm_id, openai_api_key=OPENAI_API_KEY)
    
    tools = []
    if allow_search and TAVILY_API_KEY:
        tools = [TavilySearchResults(max_results=2, tavily_api_key=TAVILY_API_KEY)]
    
    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )
    state = {"messages": query}
    response = agent.invoke(state)
    messages = response.get("messages")
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1]