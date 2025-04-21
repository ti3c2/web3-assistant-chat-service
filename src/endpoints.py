import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agent import agent_web3, run_agent
from .chroma_client import ChromaClient

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chatbot Service",
    description="API for chatbot operations",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    query: Optional[str] = None
    tokens: Optional[List[str]] = None


class ChatResponse(BaseModel):
    response: str


@app.post("/summarize/tokens")
async def get_tokens_summary(tokens: List[str]) -> ChatResponse:
    """
    Endpoint to get chatbot responses based on user tokens.

    The agent fetches information about tokens from the database and generates a response.

    NOTE: For now it is not optimized: we use ReAct agent for the simple task.
    """
    try:
        query = "What's new about tokens {}?".format(", ".join(request.tokens))
        response = await run_agent(agent_web3, query)
        return ChatResponse(response=response)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )


@app.post("/chat")
async def get_chat_response(request: ChatRequest) -> ChatResponse:
    """
    Endpoint to get chatbot responses based on user's query.

    The agent will:
    - Determine the user's intent: use semantic search over telegram posts or full text search for tokens specified in the message.
    - Analyse documents
    - Provide response with references to original posts.
    """
    if request.query is None and request.tokens is None:
        raise HTTPException(
            status_code=400, detail="Any of query or tokens is required"
        )
    try:
        query = request.query or "What's new about tokens {}?".format(
            ", ".join(request.tokens)  # pyright: ignore
        )
        response = await run_agent(agent_web3, query)
        return ChatResponse(response=response)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )
