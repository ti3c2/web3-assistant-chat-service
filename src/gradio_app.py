import asyncio

import gradio as gr

from .agent import ChatbotAgent


async def chat_response(message: str, history: list) -> str:
    chatbot = ChatbotAgent()
    response = await chatbot.process_message(message)
    return response


# Create the Gradio interface
demo = gr.ChatInterface(
    fn=chat_response,
    title="Chatbot Assistant",
    description="Ask questions about the conversation history",
    theme="default",
    # examples=[
    #     "What information do you have about recent events?",
    #     "Can you search for specific topics in the database?",
    #     "Tell me more about the available documents.",
    # ],
)

if __name__ == "__main__":
    demo.launch(share=False)
