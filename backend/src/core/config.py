from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    HOST: str = "localhost"
    PORT: int = 8000

    # LLM Settings
    LLM_MODEL: str
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 1000
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_TRACING_V2: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None

    LANGCHAIN_DEBUG: bool = True  # "logs/app.log"

    # Vector Database Settings
    VECTOR_DB_URL: str = "localhost"
    VECTOR_DB_PORT: int = 8080
    VECTOR_DB_API_KEY: Optional[str] = None
    DATA_DIR: str = "data"

    # Embedding Settings
    EMBEDDING_MODEL: str  # = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    EMBEDDING_DIMENSION: int = 1536

    # RAG Settings
    MAX_DOCUMENTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    CONTEXT_WINDOW: int = 4096

    # # Security Settings
    # API_KEY: Optional[str] = os.getenv("API_KEY")
    # JWT_SECRET: Optional[str] = os.getenv("JWT_SECRET")
    # CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")

    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None  # "logs/app.log"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
