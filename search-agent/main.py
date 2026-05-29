import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

tavily_search = TavilySearch(max_results=3)


@tool
def search(query: str) -> str:
    """
    Tool that searches over the internet
    Args:
        query: The query to search for
    Returns:
        The search results
    """
    print(f"Searching for: {query}")
    return str(tavily_search.invoke(query))


OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
llm = ChatOllama(temperature=0, model=OLLAMA_MODEL)
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hii from search agent.")
    result = agent.invoke(
        {"messages": [HumanMessage(content="Searcha any job openings in linkedin for AIML role for freshers and list their details")]}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
