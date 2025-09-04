# ai_agent.py
# Robust agent wrapper: normalizes messages, validates provider keys, and surfaces friendly errors.

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
import os
import traceback

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def _normalize_query_to_langchain_messages(query, system_prompt):
    """
    Accepts query: list[str] or list[dict] or list[pydantic models]
    Returns: list of LangChain message objects (SystemMessage/HumanMessage/AIMessage)
    """
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=str(system_prompt).strip()))

    found_user_message = False

    for item in query or []:
        # plain string -> user message
        if isinstance(item, str):
            text = item.strip()
            if text:
                messages.append(HumanMessage(content=text))
                found_user_message = True

        # pydantic model (ChatMessage) or any object with .dict()
        elif hasattr(item, "dict"):
            try:
                d = item.dict()
            except Exception:
                # fallback: ignore if dict() fails
                continue
            role = d.get("role")
            content = (d.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
                found_user_message = True
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
                found_user_message = True

        # dict literal
        elif isinstance(item, dict):
            role = item.get("role")
            content = (item.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
                found_user_message = True
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
                found_user_message = True

        else:
            # unsupported type -> ignore
            continue

    return messages, found_user_message


def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    """
    llm_id: model name string
    query: list[str] or list[dict] or pydantic ChatMessage objects
    allow_search: bool
    system_prompt: str
    provider: "Groq" or "OpenAI"
    """

    # Validate provider & keys
    if provider == "Groq":
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY missing. Set it in your .env or environment.")
    elif provider == "OpenAI":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY missing. Set it in your .env or environment.")
    else:
        raise ValueError(f"Invalid provider: {provider}")

    # Instantiate LLM
    try:
        if provider == "Groq":
            llm = ChatGroq(model=llm_id, groq_api_key=GROQ_API_KEY)
        else:  # OpenAI
            llm = ChatOpenAI(model=llm_id, openai_api_key=OPENAI_API_KEY)
    except Exception as e:
        # instantiation may fail if model not found; bubble as ValueError for backend mapping
        raise ValueError(f"LLM instantiation error for model '{llm_id}': {e}")

    # Tools
    tools = []
    if allow_search and TAVILY_API_KEY:
        tools = [TavilySearch(max_results=2, tavily_api_key=TAVILY_API_KEY)]

    # Create agent
    agent = create_react_agent(model=llm, tools=tools)

    # Normalize messages to LangChain message objects
    messages, found_user_message = _normalize_query_to_langchain_messages(query, system_prompt)

    if not found_user_message:
        raise ValueError("No user message found. Please include at least one user message in 'query'.")

    # Optional debug print:
    # debug_messages = [{"role": type(m).__name__.replace("Message","").lower(), "content": m.content} for m in messages]
    # print("DEBUG -> messages to model:", debug_messages)

    # Invoke agent and map errors into friendly exceptions
    try:
        response = agent.invoke({"messages": messages})
    except Exception as e:
        err_text = str(e).lower()
        # common Groq model-not-found signal
        if "model_not_found" in err_text or "does not exist" in err_text or "not found" in err_text:
            raise ValueError(f"Model error: the model '{llm_id}' may not exist or your API key lacks access. Original: {e}")
        # otherwise include traceback for debugging
        tb = traceback.format_exc()
        raise RuntimeError(f"Model invocation failed: {e}\n{tb}")

    result_messages = response.get("messages", [])
    ai_messages = [m.content for m in result_messages if isinstance(m, AIMessage)]
    return ai_messages[-1] if ai_messages else "⚠️ No response from model"
