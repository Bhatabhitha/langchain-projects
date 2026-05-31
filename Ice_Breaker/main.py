from typing import List
from pydantic import BaseModel, Field
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

tavily_search = TavilySearch(max_results=5)


class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """Schema for the response of the agent with answer and sources"""
    answer: str = Field(description="The answer to the question")
    sources: List[Source] = Field(default_factory=list, description="List of  sources used to answer the question")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
llm = ChatOllama(temperature=0, model=OLLAMA_MODEL)
tools = [tavily_search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hi from Ice Breaker.")
    result = agent.invoke(
        {"messages": [HumanMessage(content="Searcha any job openings in Banglore,karnataka in linkedin for AIML role for freshers and list their details")]}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
