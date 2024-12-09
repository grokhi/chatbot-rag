from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    HOST: str = "localhost"
    PORT: int = 8000

    LLM_MODEL: str = "llama3.1:8b"

    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_TRACING_V2: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None

    LANGCHAIN_DEBUG: bool = True  # "logs/app.log"

    VECTOR_DB_URL: str = "localhost"
    VECTOR_DB_PORT: int = 8080
    DATA_DIR: str = "./data/"

    # Logging Settings
    LOG_LEVEL: str = "INFO"
    # LOG_FILE: Optional[str] = None  # "logs/app.log"

    class Config:
        env_file = ".env"  # debug-only
        case_sensitive = True


settings = Settings()
