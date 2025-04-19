import logging
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
from langchain.tools import BaseTool
from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from pydantic import SecretStr

from .chroma_client import ChromaClient
from .settings import settings

logger = logging.getLogger(__name__)


class SemanticSearchTool(BaseTool):
    name: str = "semantic_search"
    description: str = "Search through knowledge base"

    def __init__(self, query_model: str = "gpt-4.1-nano"):
        super().__init__()
        self._chroma_client = ChromaClient(settings.chroma_base_url)
        self._query_model = query_model

    async def _query2chroma(self, query: str) -> str:
        prompt = "Provide the most likely answer to the following question: {query}"
        client = ChatOpenAI(
            api_key=settings.openai_api_key, model=self._query_model, temperature=0.0
        ).with_structured_output({"answer": str})
        response = await client.ainvoke(prompt.format(query=query))
        out = response["answer"]
        logger.info(
            "Transformed `{}` to `{}` for query to chroma db".format(query, out)
        )
        return out

    async def _arun(self, query: str) -> str:
        query = await self._query2chroma(query)
        results = await self._chroma_client.search(query)
        formatted_results = []
        for r in results.results:
            formatted_results.append(
                f"Document: {r.document}\n" f"From: {r.channel} at {r.datetime}\n"
            )
        logger.info("Search results: {}".format("\n".join(formatted_results)))
        return "\n".join(formatted_results)

    def _run(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("AsyncOnly does not support sync")


class ChatbotAgent:
    system_message = """\
You are a helpful web3 assistant that can answer questions about crypto, \
dexes and various activities based on your internal knowledge base.
"""

    def __init__(
        self,
        openai_api_key: SecretStr = settings.openai_api_key,
        model_name: str = settings.openai_model_name,
        temperature: float = settings.openai_temperature,
    ):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            temperature=temperature,
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.tools = [SemanticSearchTool()]
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant. Respond only in Spanish."),
                ("human", "{input}"),
                # Placeholders fill up a **list** of messages
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        self.agent = create_tool_calling_agent(
            self.llm,
            self.tools,
            self.prompt,
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
        )

    async def process_message(self, message: str) -> str:
        response = await self.agent_executor.ainvoke(dict(input=message))
        out = response["output"]
        return out


class TokenChatbotAgent:
    system_message = """\
You are a helpful web3 assistant that can advertise people activities based on their tokens, \
dexes and various activities based on your internal knowledge base.

Reference the posts with the following format:
```
<Post url>
<Relevant information>
<Exact quote>
```
"""
    user_message = """\
Here are the user's tokens: {tokens}
Here are the available posts:\n{input}
    """

    def __init__(
        self,
        openai_api_key: SecretStr = settings.openai_api_key,
        model_name: str = settings.openai_model_name,
        temperature: float = settings.openai_temperature,
    ):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            temperature=temperature,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_message),
                ("human", self.user_message),
            ]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()
        self.chroma_client = ChromaClient()

    async def ainvoke(self, input: Dict[str, Any]) -> str:
        tokens = input["tokens"]
        logger.debug(f"Querying vector store for the tokens {tokens}...")
        search_results = await self.chroma_client.search(tokens=tokens)
        formatted_posts = "\n\n".join(
            f"- Username: {result.username}\n"
            f"  Time: {result.datetime}\n"
            f"  Post URL: {result.tg_url}\n"
            f"  Content: ```\n{result.document}```\n"
            for result in search_results.results
        )
        logger.debug("Search results: {}".format(formatted_posts[:3]))
        agent_input = {"tokens": ", ".join(tokens), "input": formatted_posts}
        out = await self.chain.ainvoke(agent_input)
        return out
