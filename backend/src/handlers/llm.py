from typing import Any, Dict, Literal

from langchain_community.llms.gpt4all import GPT4All
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from src.core.config import settings
from src.core.logger import logger


class LLMHandler:

    def __init__(self, model: str = "llama3.1:8b"):
        if "groq" in model:
            self.llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, streaming=True)
        elif "llama" in model:
            self.llm = ChatOllama(model=model, temperature=0, max_tokens=1024)
        else:
            raise ValueError(
                f"Provided {model!r} was not found. Please specify correct llm model from 'llama' or 'openai' model families."
            )
