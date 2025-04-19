import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agent import TokenChatbotAgent
from .chroma_client import ChromaClient

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chatbot Service",
    description="API for chatbot operations",
    version="1.0.0",
)

# Initialize clients
chroma_client = ChromaClient()
chatbot_agent = TokenChatbotAgent()


class ChatRequest(BaseModel):
    tokens: List[str]


class ChatResponse(BaseModel):
    response: str


@app.post("/chat")
async def get_chat_response(request: ChatRequest):
    """
    Endpoint to get chatbot responses based on user tokens.

    1. The agent fetches relevant posts from vector store using tokens and sends them to LLM agent for summarization
    """
    try:
        response = await chatbot_agent.ainvoke(dict(tokens=request.tokens))
        return dict(response=response)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )
