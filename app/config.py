from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # MongoDB Configuration
    MONGO_URI: str
    DATABASE_NAME: str = "embeddings_db"
    
    # Service Configuration
    SERVICE_NAME: str = "Embeddings Service"
    VERSION: str = "1.0.0"
    
    # Model Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
