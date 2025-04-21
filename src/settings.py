import logging
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_PATH = Path(__file__).parents[2]


class ChromaConfig:
    chroma_host: str = Field(default="http://web3-assistant-vector-api")
    chroma_port: int = Field(default=6400)

    @property
    def chroma_base_url(self) -> str:
        return f"{self.chroma_host}:{self.chroma_port}"

    @property
    def chroma_search_endpoint(self) -> str:
        return f"{self.chroma_base_url}/chroma/search"


class LLMConfig:
    openai_api_key: str
    openai_api_base: str = Field(default="https://api.openai.com/v1")
    openai_model_name: str = Field(default="gpt-4o")
    openai_temperature: float = Field(default=0.3)


class ProjectSettings(BaseSettings, ChromaConfig, LLMConfig):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    log_level: int = logging.INFO


settings = ProjectSettings()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
