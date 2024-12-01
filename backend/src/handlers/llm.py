from typing import Any, Dict

from langchain_core.language_models import BaseChatModel

from backend.src.core.config import settings
from backend.src.core.logger import logger


class LLMHandler:
    """
    Singleton class to manage a single instance of the LLM (ChatOpenAI).
    """

    _instance = None

    def __new__(cls, llm: BaseChatModel = None):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(llm)
        return cls._instance

    def _initialize(self, llm: BaseChatModel):
        """
        Initialize the LLMHandler instance. This runs only once.
        """
        self._llm = llm

    @property
    def llm(self):
        return self._llm
