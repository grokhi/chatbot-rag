from typing import Any, Dict, Literal

from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq

from backend.src.core.config import settings
from backend.src.core.logger import logger

# class LLmHandler:

#     def llm_handler(self, model: Literal["local", "openai", "groq"] = "groq"):

#         if model == "groq":
#             self.llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, streaming=True)


class LLMHandler:

    def __init__(self, model: Literal["local", "openai", "groq"] = "groq"):
        if model == "groq":
            self.llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, streaming=True)


#     @property
#     def llm(self):
#         return self._llm
