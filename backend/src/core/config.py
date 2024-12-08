from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    HOST: str = "localhost"
    PORT: int = 8000

    RUN_LOCAL_LLM: bool = True
    LOCAL_LLM_HOST: str = "localhost"
    LOCAL_LLM_PORT: int = 11434
    LLAMA_MODEL: str = "llama3.1:8b"
    OPENAI_MODEL: str = ""

    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_TRACING_V2: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None

    LANGCHAIN_DEBUG: bool = False  # "logs/app.log"

    # VECTOR_DB_URL: str = "localhost"
    # VECTOR_DB_PORT: int = 8080
    DATA_DIR: str = "./data/"

    # Logging Settings
    LOG_LEVEL: str = "INFO"
    # LOG_FILE: Optional[str] = None  # "logs/app.log"

    class Config:
        env_file = ".env"  # debug-only
        case_sensitive = True


config = Settings()
