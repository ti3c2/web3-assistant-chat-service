import logging
from typing import List, Optional

import aiohttp
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from .settings import settings

logger = logging.getLogger(__name__)


class SearchQuery(BaseModel):
    query: Optional[str] = None
    tokens: Optional[List[str]] = None
    n_results: Optional[int] = None
    model_config = ConfigDict(validate_assignment=True)

    @field_validator("n_results", mode="before")
    @classmethod
    def validate_n_results(cls, n_results: int) -> int:
        return n_results or 10


class SearchResult(BaseModel):
    # Heavily exclide fields for reducing llm context
    document: str = Field(exclude=False)
    distance: float = Field(exclude=True)
    datetime: str = Field(exclude=True)
    username: str = Field(exclude=True)
    message_id: str = Field(exclude=True)
    content: str = Field(exclude=False, description="Full message content")
    chunk_id: str = Field(exclude=True)

    @computed_field
    @property
    def tg_url(self) -> str:
        return f"t.me/{self.username}/{self.message_id}"


class SearchResults(BaseModel):
    query: Optional[str] = None
    results: List[SearchResult]


class ChromaClient:
    def __init__(self):
        self.base_url = settings.chroma_base_url
        self.search_endpoint = settings.chroma_search_endpoint

    async def search(
        self,
        query: Optional[str] = None,
        tokens: Optional[List[str]] = None,
        n_results: Optional[int] = 5,
    ) -> SearchResults:
        """Perform semantic search via Chroma API."""
        search_query = SearchQuery(query=query, tokens=tokens, n_results=n_results)
        logger.debug(
            "Querying Chroma API on {}: {}".format(self.search_endpoint, search_query)
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.search_endpoint, json=search_query.model_dump()
            ) as response:
                results = await response.json()
                return SearchResults(**results)
