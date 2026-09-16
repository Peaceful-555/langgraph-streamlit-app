import os
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun


# =====================================
# Load Environment Variables
# =====================================

load_dotenv()

# =====================================
# Azure OpenAI
# =====================================

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

# =====================================
# Search Tool
# =====================================

search_tool = DuckDuckGoSearchRun()

# =====================================
# State Definition
# =====================================

class AgentState(TypedDict):
    question: str
    route: str
    search_result: str
    answer: str


# =====================================
# Router Node
# =====================================

def router_node(state: AgentState):

    question = state["question"].lower()

    search_keywords = [
        "latest",
        "today",
        "current",
        "news",
        "weather",
        "stock",
        "price",
        "ceo",
        "recent"
    ]

    use_search = any(
        keyword in question
        for keyword in search_keywords
    )

    return {
        "route": "search" if use_search else "llm"
    }


# =====================================
# Search Node
# =====================================

def search_node(state: AgentState):

    search_result = search_tool.invoke(
        state["question"]
    )

    return {
        "search_result": search_result
    }


# =====================================
# Direct LLM Node
# =====================================

def llm_node(state: AgentState):

    response = llm.invoke(
        state["question"]
    )

    return {
        "answer": response.content
    }


# =====================================
# Search + LLM Node
# =====================================

def answer_from_search_node(state: AgentState):

    prompt = f"""
    You are a helpful assistant.

    Question:
    {state['question']}

    Search Results:
    {state['search_result']}

    Use the search results to answer accurately.
    If the search results are insufficient,
    mention that clearly.
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# =====================================
# Conditional Routing Function
# =====================================

def route_question(state: AgentState):
    return state["route"]


# =====================================
# Build LangGraph
# =====================================

builder = StateGraph(AgentState)

builder.add_node("router", router_node)

builder.add_node("search", search_node)

builder.add_node("llm", llm_node)

builder.add_node(
    "search_answer",
    answer_from_search_node
)

builder.set_entry_point("router")

builder.add_conditional_edges(
    "router",
    route_question,
    {
        "search": "search",
        "llm": "llm"
    }
)

builder.add_edge(
    "search",
    "search_answer"
)

builder.add_edge(
    "search_answer",
    END
)

builder.add_edge(
    "llm",
    END
)

graph = builder.compile()

# =====================================
# Streamlit UI
# =====================================

st.set_page_config(
    page_title="LangGraph Assistant",
    page_icon="🤖"
)

st.title("🤖 LangGraph Assistant")

st.write(
    "This app demonstrates LangGraph routing with Azure OpenAI."
)

question = st.text_input(
    "Ask a question",
    placeholder="Latest news about OpenAI"
)

if st.button("Submit"):

    if question.strip():

        with st.spinner("Thinking..."):

            result = graph.invoke(
                {
                    "question": question,
                    "route": "",
                    "search_result": "",
                    "answer": ""
                }
            )

        st.subheader("Answer")

        st.write(
            result["answer"]
        )

        st.divider()

        st.subheader("Debug Information")

        st.write(
            f"Route Chosen: {result['route']}"
        )
        st.subheader("LangGraph Workflow")

        st.image(
            graph.get_graph().draw_mermaid_png()
        )