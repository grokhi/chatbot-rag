from typing import Any, Dict, Literal

from langchain_community.llms.gpt4all import GPT4All
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from backend.src.core.config import settings
from backend.src.core.logger import logger


class LLMHandler:

    def __init__(self, model: Literal["local", "openai", "groq"] = "groq"):
        if model == "groq":
            self.llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, streaming=True)
        elif model == "local":
            self.llm = ChatOllama(model="llama3.1:8b", temperature=0, max_tokens=1024)
