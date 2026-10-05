# agent.py

from typing import TypedDict

from langchain_ollama import ChatOllama
# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from tools import knowledge_base_search
from dotenv import load_dotenv
import os 
load_dotenv()


class AgentState(TypedDict):

    messages: list


llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0           ## Consistent response.
)
# HF_TOKEN = os.getenv("HF_TOKEN")
# llm = HuggingFaceEndpoint(
#     repo_id="openai/gpt-oss-120b",
#     huggingfacehub_api_token= HF_TOKEN,
#     max_new_tokens=512,
#     temperature=0.7,
# )

# model = ChatHuggingFace(llm=llm)

tools = [
    knowledge_base_search
]


llm_with_tools = llm.bind_tools(tools)


def assistant(state: AgentState):       ## LLM Node

    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": state["messages"] + [response]
    }


tool_node = ToolNode(tools)         ## tools allowed in the node.


def should_continue(state: AgentState):     ## Routing Function

    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return END


graph = StateGraph(AgentState)


graph.add_node(
    "assistant",            ## Create assistant node
    assistant
)

graph.add_node(
    "tools",                ## Creates tool node
    tool_node
)


graph.add_edge(
    START,
    "assistant"
)


graph.add_conditional_edges(
    "assistant",
    should_continue,            ## after starting, where to go next ( Conditionnal Routing)
    {
        "tools": "tools",
        END: END
    }
)


graph.add_edge(
    "tools",                   ## Going back to the assisstant after using tool.
    "assistant"
)


support_agent = graph.compile() ## Executable langraph application 

# agent.py

if __name__ == "__main__":

    question = input("User: ")

    result = support_agent.invoke({
        "messages": [
            HumanMessage(content=question)
        ]
    })

    print("\nAgent:")

    print(
        result["messages"][-1].content
    )