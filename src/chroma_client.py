import logging
from datetime import datetime
from typing import List, Optional

import aiohttp
from pydantic import BaseModel

from .settings import settings

logger = logging.getLogger(__name__)


class SearchQuery(BaseModel):
    query: Optional[str] = None
    tokens: Optional[List[str]] = None
    n_results: int = 15


class SearchResult(BaseModel):
    document: str
    distance: float
    datetime: datetime
    token_mentions: str
    username: str
    message_id: str
    chunk_id: str

    @property
    def message_idx(self) -> str:
        logger.info("Extracting message index from message ID: {}".format(self.message_id))
        items = self.message_id.split("__")
        if len(items) < 2:
            logger.warning("Invalid message ID format")
            return "000"
        return items[1]

    @property
    def tg_url(self) -> str:
        return f"t.me/{self.username}/{self.message_idx}"


class SearchResults(BaseModel):
    results: List[SearchResult]


class ChromaClient:
    def __init__(self):
        self.base_url = settings.chroma_base_url
        self.search_endpoint = settings.chroma_search_endpoint

    async def search(
        self,
        query: Optional[str] = None,
        tokens: Optional[List[str]] = None,
        n_results: int = 15,
    ) -> SearchResults:
        """Perform semantic search via Chroma API."""
        search_query = SearchQuery(query=query, tokens=tokens, n_results=n_results)
        logger.debug("Querying Chroma API on {}: {}".format(self.search_endpoint, search_query))
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.search_endpoint, json=search_query.model_dump()
            ) as response:
                results = await response.json()
                return SearchResults(**results)
