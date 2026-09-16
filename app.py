import os
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

# ----------------------------------
# Load Environment Variables
# ----------------------------------

load_dotenv()

# ----------------------------------
# Azure OpenAI
# ----------------------------------

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

search_tool = DuckDuckGoSearchRun()

# ----------------------------------
# LangGraph State
# ----------------------------------

class AgentState(TypedDict):
    question: str
    search_result: str
    answer: str


# ----------------------------------
# LangGraph Nodes
# ----------------------------------

def search_node(state: AgentState):

    result = search_tool.invoke(state["question"])

    return {
        "search_result": result
    }


def answer_node(state: AgentState):

    prompt = f"""
    Question:
    {state['question']}

    Search Results:
    {state['search_result']}

    Use the search results to answer accurately.
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# ----------------------------------
# Build Graph
# ----------------------------------

builder = StateGraph(AgentState)

builder.add_node("search", search_node)
builder.add_node("answer", answer_node)

builder.set_entry_point("search")

builder.add_edge("search", "answer")
builder.add_edge("answer", END)

graph = builder.compile()

# ----------------------------------
# Streamlit UI
# ----------------------------------

st.set_page_config(
    page_title="LangGraph Demo",
    page_icon="🤖"
)

st.title("🤖 LangGraph + Azure OpenAI")

question = st.text_input(
    "Ask anything",
    placeholder="Who is the CEO of Microsoft?"
)

if st.button("Submit"):

    if question:

        with st.spinner("Thinking..."):

            result = graph.invoke(
                {
                    "question": question,
                    "search_result": "",
                    "answer": ""
                }
            )

        st.success("Answer")

        st.write(result["answer"])