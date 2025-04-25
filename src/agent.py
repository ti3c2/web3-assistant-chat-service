import asyncio
import json
import logging
import os
from typing import Dict, List

from agents import (
    Agent,
    AsyncOpenAI,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    Runner,
    function_tool,
    trace,
)
from typing_extensions import Any, TypedDict

from .chroma_client import ChromaClient, SearchResult, SearchResults
from .settings import settings

logger = logging.getLogger(__name__)

chroma_client = ChromaClient()


@function_tool
async def full_text_search(tokens: List[str]) -> Dict[str, Any]:
    """Search across knowledge base to find documents with exact matches to the user's assets represented by tokens.

    Args:
        tokens: List of tokens to search for extracted from the user's message. Use all tokens in one query to leverage the bulk api.
    """
    results = await chroma_client.search(tokens=tokens)
    logger.debug( "Full text search results for tokens {}: {}".format(tokens, json.dumps(results.model_dump(), indent=2, ensure_ascii=False))) # fmt: skip
    return results.model_dump()


@function_tool
async def semantic_search(query: str) -> Dict[str, Any]:
    """Search across knowledge base to find documents with the information relevant to the user's query.
    Use it to extract information about latest news

    Args:
        query: A query in direct form that will fetch information relevant to the user's query.
            - Use the direct query:
                * if users asks "What are the available trading courses?" then your query is "trading courses"
            - Use the language of the query
    """
    results = await chroma_client.search(query=query)
    logger.debug("Semantic search results for `{}`: {}".format(query, json.dumps(results.model_dump(), indent=2, ensure_ascii=False))) # fmt: skip
    return results.model_dump()


agent_web3 = Agent(
    name="Web3 Agent",
    instructions="""\
You are a helpful Web3 assistant that can advertise people activities based on their tokens, \
dexes and various activities based on your internal knowledge base.

You are going to talk to users about their portfolio and Web3 activities.

Talk in engaging and funny style.

Additional instructions:
- Always add tg_url as a link to original posts. Put it exactly as specified, like `t.me/user/123`
- Use quotes from original posts, especially for numbers.
- If you see some html formatting, change it to normal text.

Note Web3 Terminology:
- *Farming* is when users to lock their cryptocurrency tokens for a set period to earn rewards for their tokens
""",
    model=OpenAIChatCompletionsModel(
        model=settings.openai_model_name,
        openai_client=AsyncOpenAI(
            api_key=settings.openai_api_key, base_url=settings.openai_api_base
        ),
    ),
    tools=[full_text_search, semantic_search],
    model_settings=ModelSettings(),
)


async def run_agent(agent: Agent, query: str) -> str:
    result = await Runner.run(agent, query)
    return result.final_output


async def main():
    import argparse

    parser = argparse.ArgumentParser("Test single query for Web3 Agent")
    parser.add_argument("-q", "--query", type=str, help="Search query")
    # parser.add_argument("-ch", "--chroma-host", type=str, default="http://localhost", help="Host for ChromaDB") # fmt: skip
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging") # fmt: skip
    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    with trace(workflow_name="Web3 Agent"):
        result = await Runner.run(agent_web3, args.query)
        print("\n\nQUERY: {}".format(args.query))
        print("OUTPUT:\n", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
