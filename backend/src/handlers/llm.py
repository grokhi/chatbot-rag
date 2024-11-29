from typing import Any, Dict

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAI

from backend.src.core.config import settings
from backend.src.core.logger import logger


class LLMHandler:
    """
    Singleton class to manage a single instance of the LLM (ChatOpenAI).
    """

    _instance = None  # Class-level private instance

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            # Create a new instance if none exists
            cls._instance = super().__new__(cls)
            cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(self, model: str):
        """
        Initialize the LLMHandler instance. This runs only once.
        """
        # self.llm = ChatOpenAI(model=model)
        self.llm = ChatGroq(model="llama-3.1-70b-versatile")

    def get_llm(self):
        """
        Returns the LLM instance.
        """
        return self.llm
