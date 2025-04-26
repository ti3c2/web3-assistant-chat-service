# Web3 Assistant Chat Service

A chatbot service that leverages LLM capabilities to provide intelligent responses about web3/crypto topics using a vector database backend.

## Components

## API Endpoints
Swagger UI: http://localhost:4400/docs

### POST `/chat`
Token-based chat interaction endpoint.

**Request Body:**
```json
{
    "tokens": ["token1", "token2"]
}
```

**Response:**
```json
{
    "response": "Detailed analysis and recommendations based on tokens"
}
```

## Usage

### Starting the Service

1. Set up environment variables using `.env` file
2. Make sure `web3-assistant-network` Docker network is created
3. Launch app: `docker-compose up`

### Available CLI
Test agent with query: `python -m src.agent -q "Are there any trading courses?"`. Run from inside the container.

## Code Diagram
```mermaid
classDiagram
    direction TB

    class ProjectSettings {
        +chroma_host: str
        +chroma_port: int
        +openai_api_key: str
        +openai_api_base: str
        +openai_model_name: str
        +openai_temperature: float
        +chroma_base_url: str
        +chroma_search_endpoint: str
    }

    class ChromaClient {
        +base_url: str
        +search_endpoint: str
        +search(query: str, tokens: List[str], n_results: int)
    }

    class SearchQuery {
        +query: Optional[str]
        +tokens: Optional[List[str]]
        +n_results: Optional[int]
    }

    class SearchResult {
        +document: str
        +distance: float
        +datetime: str
        +username: str
        +message_id: str
        +content: str
        +chunk_id: str
        +tg_url: str
    }

    class SearchResults {
        +query: Optional[str]
        +results: List[SearchResult]
    }

    class Agent {
        +name: str
        +instructions: str
        +model: OpenAIChatCompletionsModel
        +tools: List[Tool]
    }

    class ChatRequest {
        +query: Optional[str]
        +tokens: Optional[List[str]]
    }

    class ChatResponse {
        +response: str
    }

    class FastAPI {
        +POST /summarize/tokens
        +POST /chat
    }

    ProjectSettings --> ChromaClient
    ProjectSettings --> Agent

    ChromaClient --> SearchQuery
    ChromaClient --> SearchResults
    SearchResults --> SearchResult
    ChatRequest --> FastAPI
    ChatResponse <-- FastAPI
    FastAPI --> Agent
    Agent --> ChromaClient
```
