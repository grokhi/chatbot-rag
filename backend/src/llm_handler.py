from typing import Any, Dict

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAI

from backend.src.config import settings

from .logger import logger


class LLMHandler:
    """Handler for interacting with the language model."""

    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: str = None):
        """
        Initialize the LLM handler with the specified model and API key.

        Args:
            model_name (str): The name of the model to use.
            api_key (str): The API key for authenticating with the LLM service.
        """
        self.model_name = model_name
        self.api_key = api_key

        # self.llm = ChatOpenAI(model=model_name, api_key=api_key)
        self.llm = ChatGroq(
            model="llama-3.1-70b-versatile",
        )

    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the language model based on the given prompt.

        Args:
            prompt (str): The input prompt for the language model.
            **kwargs: Additional parameters for the LLM API call.

        Returns:
            str: The generated response from the language model.
        """
        try:
            # Example using OpenAI's API
            # response = openai.Completion.create(
            #     engine=self.model_name,
            #     prompt=prompt,
            #     max_tokens=150,
            #     n=1,
            #     stop=None,
            #     temperature=0.7,
            #     **kwargs
            # )
            # return response.choices[0].text.strip()

            # Placeholder for actual LLM call
            response = "This is a placeholder response."
            return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "Error generating response."


# # Example usage
# if __name__ == "__main__":
#     handler = LLMHandler(api_key=se")
#     prompt = "What is the capital of France?"
#     response = handler.generate_response(prompt)
#     print(response)
